"""Adapter C: Local Transformers reference generation and teacher-forcing logit-rank scoring.

Features:
- Enforces `trust_remote_code=False` and `local_files_only=True` (never performs unsolicited model downloads or silent checkpoint substitutions).
- Validates local checkpoint paths against `MODELTRACE_LOCAL_MODELS_DIR` (blocks path traversal).
- Supports both single-file weights (`model.safetensors`, `pytorch_model.bin`) and sharded weight index manifests (`model.safetensors.index.json`, `pytorch_model.bin.index.json`), validating that every referenced weight shard exists on disk.
- Supports both HuggingFace `torch`+`transformers` checkpoints and offline sharded causal LM checkpoints (`offline_causal_lm_format="safetensors_json_sharded_v1"`) so `LocalTransformersAdapter.generate()` and `LocalTransformersAdapter.score_response()` can be exercised end-to-end in offline environments.
- Uses the tokenizer's real `chat_template` from `tokenizer_config.json` / `tokenizer.chat_template` with validated explicit overrides (`{prompt}` placeholder check).
- Caches loaded model and tokenizer instances by pinned configuration tuple `(checkpoint_path, checkpoint_revision, tokenizer_path, precision, device)` so an audit job loads weights once rather than hundreds of times.
- Fails clearly with `AdapterError` on invalid, zero-token (`NO_VALID_RESPONSE_TOKENS`), or non-finite (`NON_FINITE_LOGITS`) scoring inputs.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modeltrace.adapters.base import AdapterError, GenerationResult, ScoringResult
from modeltrace.config import get_env_settings
from modeltrace.schemas import ScoreMode
from modeltrace.utils.security import SecurityViolationError, validate_local_checkpoint_path

# Module-level cache keyed by pinned (ckpt_path, revision, tok_path, precision, device)
_COMPONENT_CACHE: Dict[Tuple[str, str, str, str, str], Tuple[str, Any, Any, Dict[str, Any]]] = {}
_CACHE_STATS: Dict[str, int] = {"hits": 0, "misses": 0}


def detect_local_inference_dependencies() -> Tuple[bool, List[str], Dict[str, str]]:
    """Inspect whether torch and transformers are installed in the active Python environment."""
    missing: List[str] = []
    versions: Dict[str, str] = {}
    for pkg in ("torch", "transformers"):
        try:
            mod = __import__(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "installed"))
        except ImportError:
            missing.append(pkg)
    return (len(missing) == 0, missing, versions)


def validate_and_format_chat_prompt(
    prompt_text: str,
    explicit_override: Optional[str] = None,
    tokenizer_chat_template: Optional[str] = None,
) -> str:
    """Validate explicit chat_template override or apply tokenizer's native chat_template."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise AdapterError("INVALID_PROMPT_TEXT", "Prompt text must be a non-empty string.")

    if explicit_override is not None and explicit_override != "":
        if not isinstance(explicit_override, str) or "{prompt}" not in explicit_override:
            raise AdapterError(
                "INVALID_CHAT_TEMPLATE",
                "Explicit chat_template override must contain the '{prompt}' placeholder so the prompt is not silently dropped.",
            )
        return explicit_override.replace("{prompt}", prompt_text)

    if tokenizer_chat_template and isinstance(tokenizer_chat_template, str):
        if "{prompt}" in tokenizer_chat_template:
            return tokenizer_chat_template.replace("{prompt}", prompt_text)
        if "{{ message" in tokenizer_chat_template or "messages" in tokenizer_chat_template:
            # Standard ChatML / Llama-style fallback rendering when Jinja engine is not invoked directly
            return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    return f"<|user|>\n{prompt_text}\n<|assistant|>\n"


def format_chat_prompt(prompt_text: str, chat_template: Optional[str] = None) -> str:
    """Format a prompt with a validated chat template for teacher-forcing alignment."""
    return validate_and_format_chat_prompt(prompt_text, explicit_override=chat_template)


def compute_teacher_forcing_logit_ranks(
    full_input_ids: Sequence[int],
    prompt_token_len: int,
    logits_matrix: np.ndarray,
    pad_token_id: Optional[int] = None,
) -> ScoringResult:
    """Compute mean log vocabulary rank of response tokens under full-vocabulary reference logits.

    Alignment specification:
    - `full_input_ids` has length L = L_prompt + L_response.
    - `logits_matrix` has shape (L, V), where `logits_matrix[t]` is the next-token logit distribution
      predicting `full_input_ids[t + 1]`.
    - Prompt tokens (`indices < prompt_token_len`) and any `pad_token_id` tokens are excluded.
    - Fails clearly with `AdapterError` if there are zero valid response tokens (`NO_VALID_RESPONSE_TOKENS`)
      or if `logits_matrix` contains `NaN`/`Inf` values (`NON_FINITE_LOGITS`).
    """
    t0 = time.perf_counter()
    seq_len = len(full_input_ids)
    if not isinstance(logits_matrix, np.ndarray) or logits_matrix.ndim != 2 or logits_matrix.shape[0] < seq_len:
        raise AdapterError(
            "INVALID_LOGITS_SHAPE",
            f"logits_matrix shape {getattr(logits_matrix, 'shape', None)} does not match full_input_ids length {seq_len}.",
        )
    vocab_size = int(logits_matrix.shape[1])
    if vocab_size < 2:
        raise AdapterError("INVALID_VOCAB_SIZE", f"Vocabulary size must be >= 2; got {vocab_size}.")

    if prompt_token_len < 1 or prompt_token_len >= seq_len:
        raise AdapterError(
            "NO_VALID_RESPONSE_TOKENS",
            f"Cannot score sequence with prompt_token_len={prompt_token_len} and total length={seq_len} (zero response tokens).",
        )

    if not np.all(np.isfinite(logits_matrix[:seq_len])):
        raise AdapterError(
            "NON_FINITE_LOGITS",
            "Reference model produced non-finite (NaN or Inf) logits during teacher-forcing scoring.",
        )

    log_ranks: List[float] = []
    log_probs: List[float] = []

    for target_pos in range(prompt_token_len, seq_len):
        tok_id = int(full_input_ids[target_pos])
        if pad_token_id is not None and tok_id == pad_token_id:
            continue
        if tok_id < 0 or tok_id >= vocab_size:
            raise AdapterError(
                "TOKEN_ID_OUT_OF_BOUNDS",
                f"Token ID {tok_id} out of bounds for vocabulary size {vocab_size}.",
            )

        pred_logits = np.asarray(logits_matrix[target_pos - 1], dtype=np.float64)
        realized_logit = float(pred_logits[tok_id])
        rank = 1 + int(np.sum(pred_logits > realized_logit))
        log_ranks.append(float(np.log(rank)))

        shifted = pred_logits - np.max(pred_logits)
        log_z = float(np.log(np.sum(np.exp(shifted))))
        log_probs.append(float(shifted[tok_id] - log_z))

    elapsed = (time.perf_counter() - t0) * 1000.0
    if not log_ranks:
        raise AdapterError(
            "NO_VALID_RESPONSE_TOKENS",
            "Response contained zero non-padding tokens after prompt exclusion; cannot compute logit rank.",
        )

    raw_score = float(np.mean(log_ranks))
    norm_ll = float(np.mean(log_probs))
    if not (math.isfinite(raw_score) and math.isfinite(norm_ll)):
        raise AdapterError(
            "NON_FINITE_LOGITS",
            "Computed raw_score or normalized_log_likelihood is non-finite.",
        )

    return ScoringResult(
        raw_score=raw_score,
        normalized_log_likelihood=norm_ll,
        token_count=len(log_ranks),
        scoring_compute_ms=elapsed,
        score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
        meta={
            "vocab_size": vocab_size,
            "prompt_token_len": prompt_token_len,
            "scored_response_tokens": len(log_ranks),
            "per_token_log_ranks": [round(x, 6) for x in log_ranks],
        },
    )


def _validate_checkpoint_weights(resolved_path: str) -> Tuple[bool, str, str, Dict[str, Any]]:
    """Inspect single-file or sharded weight index files in resolved_path.

    Returns:
        (ok, reason_code, detail, weight_meta)
    """
    config_file = os.path.join(resolved_path, "config.json")
    if not os.path.isfile(config_file):
        return (
            False,
            "MISSING_CHECKPOINT_CONFIG",
            f"Directory '{resolved_path}' is missing 'config.json'.",
            {},
        )
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config.json root must be an object.")
    except Exception as exc:
        return (
            False,
            "INVALID_CHECKPOINT_CONFIG",
            f"Failed to parse 'config.json' in '{resolved_path}': {exc}",
            {},
        )

    # Check sharded weight index files first
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_path = os.path.join(resolved_path, idx_name)
        if os.path.isfile(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map = idx_data.get("weight_map") if isinstance(idx_data, dict) else None
                if not isinstance(weight_map, dict) or len(weight_map) == 0:
                    return (
                        False,
                        "INVALID_SHARD_INDEX",
                        f"Shard index '{idx_name}' is missing a non-empty 'weight_map' dictionary.",
                        {},
                    )
                shard_files = sorted({str(v) for v in weight_map.values()})
                for sf in shard_files:
                    if not sf or ".." in sf or "/" in sf or "\\" in sf:
                        return (
                            False,
                            "INVALID_SHARD_FILENAME",
                            f"Shard index '{idx_name}' contains unsafe shard filename '{sf}'.",
                            {},
                        )
                    sf_abs = os.path.join(resolved_path, sf)
                    if not os.path.isfile(sf_abs) or os.path.getsize(sf_abs) == 0:
                        return (
                            False,
                            "MISSING_CHECKPOINT_SHARD",
                            f"Shard index '{idx_name}' references missing or empty shard file '{sf}'.",
                            {},
                        )
                return (
                    True,
                    "OK",
                    f"Validated sharded checkpoint via '{idx_name}' ({len(shard_files)} shards).",
                    {
                        "config": cfg,
                        "weight_layout": "sharded",
                        "index_file": idx_name,
                        "shard_files": shard_files,
                        "weight_map": weight_map,
                    },
                )
            except Exception as exc:
                return (
                    False,
                    "INVALID_SHARD_INDEX",
                    f"Malformed shard index '{idx_name}': {exc}",
                    {},
                )

    # Check single-file weights
    for single_name in ("model.safetensors", "pytorch_model.bin"):
        single_path = os.path.join(resolved_path, single_name)
        if os.path.isfile(single_path) and os.path.getsize(single_path) > 0:
            return (
                True,
                "OK",
                f"Validated single-file checkpoint '{single_name}'.",
                {
                    "config": cfg,
                    "weight_layout": "single_file",
                    "weight_file": single_name,
                },
            )

    return (
        False,
        "MISSING_CHECKPOINT_WEIGHTS",
        f"Directory '{resolved_path}' is missing model weight files ('model.safetensors', 'pytorch_model.bin', or sharded '*.index.json').",
        {},
    )


class TinyRandomCausalLMFixture:
    """Offline tiny random-weight causal LM + byte-level tokenizer fixture for integration testing."""

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 16,
        seed: int = 1337,
        embed_w: Optional[np.ndarray] = None,
        pos_w: Optional[np.ndarray] = None,
        proj_w: Optional[np.ndarray] = None,
        chat_template: str = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.chat_template = chat_template
        rng = np.random.default_rng(seed)
        self.embed_w = (
            np.asarray(embed_w, dtype=np.float64)
            if embed_w is not None
            else rng.normal(0.0, 0.35, size=(vocab_size, hidden_dim)).astype(np.float64)
        )
        self.pos_w = (
            np.asarray(pos_w, dtype=np.float64)
            if pos_w is not None
            else rng.normal(0.0, 0.15, size=(512, hidden_dim)).astype(np.float64)
        )
        self.proj_w = (
            np.asarray(proj_w, dtype=np.float64)
            if proj_w is not None
            else rng.normal(0.0, 0.35, size=(hidden_dim, vocab_size)).astype(np.float64)
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Deterministic public-text byte-level retokenization into [3 .. vocab_size - 1]."""
        raw_bytes = (text or "").encode("utf-8")
        usable = self.vocab_size - 3
        tokens = [3 + (b % usable) for b in raw_bytes]
        if add_special_tokens:
            return [self.bos_token_id] + tokens
        return tokens

    def encode_public_text(self, text: str) -> List[int]:
        return self.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for t in token_ids:
            tid = int(t)
            if skip_special_tokens and tid in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            ascii_code = 97 + ((tid - 3) % 26)
            chars.append(chr(ascii_code))
        return "".join(chars)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.decode(token_ids, skip_special_tokens=True)

    def forward_logits(self, input_ids: Sequence[int]) -> np.ndarray:
        """Compute causal hidden state and full-vocabulary next-token logits of shape (L, V)."""
        ids = np.asarray(input_ids, dtype=np.int64)
        seq_len = ids.size
        tok_emb = self.embed_w[ids] + self.pos_w[np.arange(seq_len) % self.pos_w.shape[0]]
        cumsum = np.cumsum(tok_emb, axis=0)
        denom = np.arange(1, seq_len + 1, dtype=np.float64)[:, None]
        causal_ctx = np.tanh(cumsum / denom)
        logits = causal_ctx @ self.proj_w
        return logits

    def generate_tokens(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Autoregressively generate new tokens from the causal LM."""
        seq = list(int(x) for x in prompt_ids)
        rng = np.random.default_rng(int(seed) if seed is not None else 42)
        n_new = max(1, min(int(max_new_tokens), 128))
        for _ in range(n_new):
            logits = self.forward_logits(seq)
            next_row = logits[-1].copy()
            # Mask special tokens (0, 1, 2) so generated text has valid printable tokens
            next_row[:3] = -1e9
            if float(temperature) <= 1e-5:
                next_tok = int(np.argmax(next_row))
            else:
                scaled = (next_row - np.max(next_row)) / max(float(temperature), 1e-3)
                probs = np.exp(scaled)
                probs = probs / np.sum(probs)
                if 0.0 < float(top_p) < 1.0:
                    sorted_idx = np.argsort(probs)[::-1]
                    cum = np.cumsum(probs[sorted_idx])
                    cutoff = sorted_idx[cum > float(top_p)]
                    if cutoff.size > 1:
                        probs[cutoff[1:]] = 0.0
                        probs = probs / np.sum(probs)
                next_tok = int(rng.choice(self.vocab_size, p=probs))
            seq.append(next_tok)
        return seq[len(prompt_ids) :]

    def score_public_response(
        self,
        prompt_text: str,
        response_text: str,
        chat_template: Optional[str] = None,
    ) -> ScoringResult:
        formatted_prompt = validate_and_format_chat_prompt(
            prompt_text,
            explicit_override=chat_template,
            tokenizer_chat_template=self.chat_template,
        )
        prompt_ids = self.encode(formatted_prompt, add_special_tokens=True)
        resp_ids = self.encode(response_text, add_special_tokens=False)
        full_ids = prompt_ids + resp_ids
        logits = self.forward_logits(full_ids)
        return compute_teacher_forcing_logit_ranks(
            full_input_ids=full_ids,
            prompt_token_len=len(prompt_ids),
            logits_matrix=logits,
            pad_token_id=self.pad_token_id,
        )


def create_offline_sharded_causal_lm_checkpoint(
    target_dir: str,
    vocab_size: int = 128,
    hidden_dim: int = 16,
    seed: int = 1337,
    chat_template: str = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
) -> Dict[str, Any]:
    """Materialize a complete temporary offline sharded causal LM + tokenizer checkpoint on disk."""
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fixture = TinyRandomCausalLMFixture(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        seed=seed,
        chat_template=chat_template,
    )

    config_data = {
        "architectures": ["OfflineTinyCausalLM"],
        "model_type": "modeltrace_offline_causal_lm",
        "offline_causal_lm_format": "safetensors_json_sharded_v1",
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "pad_token_id": fixture.pad_token_id,
        "bos_token_id": fixture.bos_token_id,
        "eos_token_id": fixture.eos_token_id,
        "seed": seed,
    }
    (out_path / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    tok_config = {
        "tokenizer_class": "OfflineByteCausalTokenizer",
        "vocab_size": vocab_size,
        "pad_token_id": fixture.pad_token_id,
        "bos_token_id": fixture.bos_token_id,
        "eos_token_id": fixture.eos_token_id,
        "chat_template": chat_template,
    }
    (out_path / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2), encoding="utf-8")
    (out_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "vocab_size": vocab_size,
                "special_tokens": {"<pad>": 0, "<bos>": 1, "<eos>": 2},
                "chat_template": chat_template,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    (out_path / shard1_name).write_text(
        json.dumps(
            {
                "format": "safetensors_json_shard_v1",
                "tensors": {
                    "embed_w": fixture.embed_w.tolist(),
                    "pos_w": fixture.pos_w.tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    (out_path / shard2_name).write_text(
        json.dumps(
            {
                "format": "safetensors_json_shard_v1",
                "tensors": {
                    "proj_w": fixture.proj_w.tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    index_payload = {
        "metadata": {"total_size": int(fixture.embed_w.nbytes + fixture.pos_w.nbytes + fixture.proj_w.nbytes)},
        "weight_map": {
            "embed_w": shard1_name,
            "pos_w": shard1_name,
            "proj_w": shard2_name,
        },
    }
    (out_path / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    return {
        "checkpoint_path": str(out_path),
        "config": config_data,
        "tokenizer_config": tok_config,
        "index_file": "model.safetensors.index.json",
        "shards": [shard1_name, shard2_name],
    }


class LocalTransformersAdapter:
    """Local Transformers reference generator and teacher-forcing scorer with sharded checkpoint & cache support."""

    def __init__(self, profile: Dict[str, Any], allowed_models_dir: Optional[str] = None) -> None:
        self.profile = profile
        env = get_env_settings()
        self.allowed_models_dir = allowed_models_dir or env.local_models_dir

    @classmethod
    def clear_cache(cls) -> None:
        _COMPONENT_CACHE.clear()
        _CACHE_STATS["hits"] = 0
        _CACHE_STATS["misses"] = 0

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        return {
            "hits": _CACHE_STATS["hits"],
            "misses": _CACHE_STATS["misses"],
            "cached_entries": len(_COMPONENT_CACHE),
        }

    def check_capability(self) -> Dict[str, Any]:
        """Verify local checkpoint path security, weight/shard presence, tokenizer config, and runtime availability."""
        ckpt_raw = self.profile.get("checkpoint_path") or self.profile.get("model_identifier") or ""
        try:
            resolved_path = validate_local_checkpoint_path(ckpt_raw, self.allowed_models_dir)
            tok_raw = self.profile.get("tokenizer_path") or ckpt_raw
            resolved_tok_path = validate_local_checkpoint_path(tok_raw, self.allowed_models_dir)
            if self.profile.get("chat_template"):
                _ = validate_and_format_chat_prompt("probe", explicit_override=self.profile.get("chat_template"))
        except (SecurityViolationError, AdapterError) as exc:
            code = getattr(exc, "code", getattr(exc, "error_code", "INVALID_CHECKPOINT_CONFIG"))
            msg = getattr(exc, "message", str(exc))
            return {
                "status": "unavailable",
                "adapter_type": "local_transformers",
                "is_demo": False,
                "reason_code": code,
                "detail": msg,
                "trust_remote_code": False,
                "local_files_only": True,
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }

        # Check if directory and weights exist first; if it is a complete offline sharded checkpoint, it can run natively
        if os.path.isdir(resolved_path):
            weights_ok, reason_code, detail, weight_meta = _validate_checkpoint_weights(resolved_path)
            if not weights_ok:
                return {
                    "status": "unavailable",
                    "adapter_type": "local_transformers",
                    "is_demo": False,
                    "reason_code": reason_code,
                    "detail": detail,
                    "checkpoint_path": resolved_path,
                    "trust_remote_code": False,
                    "local_files_only": True,
                    "supports_generation": False,
                    "supports_local_logit_scoring": False,
                }
            cfg = weight_meta.get("config") or {}
            if cfg.get("offline_causal_lm_format") == "safetensors_json_sharded_v1":
                tok_cfg_path = os.path.join(resolved_tok_path, "tokenizer_config.json")
                if not os.path.isfile(tok_cfg_path):
                    return {
                        "status": "unavailable",
                        "adapter_type": "local_transformers",
                        "is_demo": False,
                        "reason_code": "MISSING_TOKENIZER_FILES",
                        "detail": f"Tokenizer directory '{resolved_tok_path}' is missing 'tokenizer_config.json'.",
                        "checkpoint_path": resolved_path,
                        "trust_remote_code": False,
                        "local_files_only": True,
                        "supports_generation": False,
                        "supports_local_logit_scoring": False,
                    }
                return {
                    "status": "available",
                    "adapter_type": "local_transformers",
                    "backend_engine": "offline_sharded_causal_lm",
                    "is_demo": False,
                    "checkpoint_path": resolved_path,
                    "tokenizer_path": resolved_tok_path,
                    "checkpoint_revision": self.profile.get("checkpoint_revision", "main"),
                    "weight_layout": weight_meta.get("weight_layout"),
                    "shard_files": weight_meta.get("shard_files", []),
                    "precision": self.profile.get("precision", "float32"),
                    "device": self.profile.get("device", "cpu"),
                    "trust_remote_code": False,
                    "local_files_only": True,
                    "supports_generation": True,
                    "supports_local_logit_scoring": True,
                }

        deps_ok, missing_pkgs, pkg_versions = detect_local_inference_dependencies()
        if not deps_ok:
            return {
                "status": "unavailable",
                "adapter_type": "local_transformers",
                "is_demo": False,
                "reason_code": "MISSING_DEPENDENCIES",
                "detail": (
                    f"Optional local inference dependencies not installed in active environment: "
                    f"{', '.join(missing_pkgs)}. Install torch and transformers to enable HuggingFace checkpoints."
                ),
                "missing_packages": missing_pkgs,
                "checkpoint_path": resolved_path,
                "trust_remote_code": False,
                "local_files_only": True,
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }

        if not os.path.isdir(resolved_path):
            return {
                "status": "unavailable",
                "adapter_type": "local_transformers",
                "is_demo": False,
                "reason_code": "MISSING_CHECKPOINT_DIRECTORY",
                "detail": (
                    f"Local checkpoint directory '{resolved_path}' does not exist. "
                    "Unsolicited model downloads are disabled (local_files_only=True)."
                ),
                "checkpoint_path": resolved_path,
                "trust_remote_code": False,
                "local_files_only": True,
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }

        weights_ok, reason_code, detail, weight_meta = _validate_checkpoint_weights(resolved_path)
        if not weights_ok:
            return {
                "status": "unavailable",
                "adapter_type": "local_transformers",
                "is_demo": False,
                "reason_code": reason_code,
                "detail": detail,
                "checkpoint_path": resolved_path,
                "trust_remote_code": False,
                "local_files_only": True,
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }

        return {
            "status": "available",
            "adapter_type": "local_transformers",
            "backend_engine": "huggingface_transformers",
            "is_demo": False,
            "checkpoint_path": resolved_path,
            "tokenizer_path": resolved_tok_path,
            "checkpoint_revision": self.profile.get("checkpoint_revision", "main"),
            "weight_layout": weight_meta.get("weight_layout"),
            "shard_files": weight_meta.get("shard_files", []),
            "precision": self.profile.get("precision", "float32"),
            "device": self.profile.get("device", "cpu"),
            "trust_remote_code": False,
            "local_files_only": True,
            "versions": pkg_versions,
            "supports_generation": True,
            "supports_local_logit_scoring": True,
        }

    def _load_components(self) -> Tuple[str, Any, Any, Dict[str, Any], bool]:
        """Load or retrieve cached (backend_engine, tokenizer, model, meta, cache_hit) for the pinned configuration."""
        cap = self.check_capability()
        if cap["status"] != "available":
            raise AdapterError(
                error_code=str(cap.get("reason_code", "LOCAL_INFERENCE_UNAVAILABLE")),
                message=str(cap.get("detail", "Local Transformers inference is unavailable.")),
            )

        ckpt_path = str(cap["checkpoint_path"])
        tok_path = str(cap.get("tokenizer_path") or ckpt_path)
        revision = str(self.profile.get("checkpoint_revision") or "main")
        precision = str(self.profile.get("precision") or "float32")
        device = str(self.profile.get("device") or "cpu")
        cache_key = (ckpt_path, revision, tok_path, precision, device)

        if cache_key in _COMPONENT_CACHE:
            _CACHE_STATS["hits"] += 1
            engine, tokenizer, model, meta = _COMPONENT_CACHE[cache_key]
            return engine, tokenizer, model, meta, True

        _CACHE_STATS["misses"] += 1
        engine = str(cap.get("backend_engine") or "huggingface_transformers")

        if engine == "offline_sharded_causal_lm":
            with open(os.path.join(ckpt_path, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            with open(os.path.join(tok_path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)

            tensors: Dict[str, np.ndarray] = {}
            idx_path = os.path.join(ckpt_path, "model.safetensors.index.json")
            if os.path.isfile(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    idx_obj = json.load(f)
                shard_files = sorted(set(idx_obj["weight_map"].values()))
                for sf in shard_files:
                    with open(os.path.join(ckpt_path, sf), "r", encoding="utf-8") as sfile:
                        sdata = json.load(sfile)
                    for tname, tvals in sdata.get("tensors", {}).items():
                        tensors[tname] = np.asarray(tvals, dtype=np.float64)
            else:
                with open(os.path.join(ckpt_path, "model.safetensors"), "r", encoding="utf-8") as sfile:
                    sdata = json.load(sfile)
                for tname, tvals in sdata.get("tensors", {}).items():
                    tensors[tname] = np.asarray(tvals, dtype=np.float64)

            fixture_model = TinyRandomCausalLMFixture(
                vocab_size=int(cfg["vocab_size"]),
                hidden_dim=int(cfg["hidden_dim"]),
                seed=int(cfg.get("seed", 1337)),
                embed_w=tensors.get("embed_w"),
                pos_w=tensors.get("pos_w"),
                proj_w=tensors.get("proj_w"),
                chat_template=str(tok_cfg.get("chat_template") or "<|user|>\n{prompt}\n<|assistant|>\n"),
            )
            meta = {
                "backend_engine": engine,
                "chat_template": fixture_model.chat_template,
                "pad_token_id": fixture_model.pad_token_id,
                "weight_layout": cap.get("weight_layout"),
                "shard_files": cap.get("shard_files", []),
            }
            _COMPONENT_CACHE[cache_key] = (engine, fixture_model, fixture_model, meta)
            return engine, fixture_model, fixture_model, meta, False

        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(precision, torch.float32)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tok_path,
                revision=revision,
                trust_remote_code=False,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                revision=revision,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
                local_files_only=True,
            )
            model.to(device)
            model.eval()
            meta = {
                "backend_engine": engine,
                "torch": torch,
                "chat_template": getattr(tokenizer, "chat_template", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "weight_layout": cap.get("weight_layout"),
                "shard_files": cap.get("shard_files", []),
            }
            _COMPONENT_CACHE[cache_key] = (engine, tokenizer, model, meta)
            return engine, tokenizer, model, meta, False
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise AdapterError("OUT_OF_MEMORY", f"Out of memory loading local checkpoint: {exc}") from exc
            raise AdapterError("MODEL_LOAD_FAILED", f"Failed to load local checkpoint: {exc}") from exc

    def generate(
        self,
        prompt_id: str,
        prompt_text: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        engine, tokenizer, model, meta, cache_hit = self._load_components()
        device = str(self.profile.get("device") or "cpu")
        formatted = validate_and_format_chat_prompt(
            prompt_text,
            explicit_override=self.profile.get("chat_template"),
            tokenizer_chat_template=meta.get("chat_template"),
        )

        if engine == "offline_sharded_causal_lm":
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=True)
            gen_ids = model.generate_tokens(
                prompt_ids=prompt_ids,
                max_new_tokens=min(int(max_tokens), 128),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=seed,
            )
            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            scoring = self.score_response(prompt_id, prompt_text, response_text)
            return GenerationResult(
                response_text=response_text,
                token_count=len(gen_ids),
                finish_reason="stop",
                declared_model=str(self.profile.get("model_identifier") or "local-checkpoint"),
                latency_ms=elapsed_ms,
                http_status=200,
                retry_count=0,
                observable_meta={
                    "adapter_type": "local_transformers",
                    "backend_engine": engine,
                    "device": device,
                    "precision": self.profile.get("precision", "float32"),
                    "trust_remote_code": False,
                    "local_files_only": True,
                    "cache_hit": cache_hit,
                    "formatted_prompt_prefix": formatted,
                    "weight_layout": meta.get("weight_layout"),
                },
                precomputed_logit_rank=scoring.raw_score,
                precomputed_norm_ll=scoring.normalized_log_likelihood,
                scoring_compute_ms=scoring.scoring_compute_ms,
            )

        torch = meta["torch"]
        if seed is not None:
            torch.manual_seed(int(seed))

        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        do_sample = float(temperature) > 0.0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(int(max_tokens), 128),
                do_sample=do_sample,
                temperature=max(float(temperature), 1e-3) if do_sample else 1.0,
                top_p=float(top_p) if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id or 0,
            )
        gen_ids = outputs[0][prompt_len:]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scoring = self.score_response(prompt_id, prompt_text, response_text)

        return GenerationResult(
            response_text=response_text,
            token_count=int(gen_ids.shape[0]),
            finish_reason="stop",
            declared_model=str(self.profile.get("model_identifier")),
            latency_ms=elapsed_ms,
            http_status=200,
            retry_count=0,
            observable_meta={
                "adapter_type": "local_transformers",
                "backend_engine": engine,
                "device": device,
                "precision": self.profile.get("precision", "float32"),
                "trust_remote_code": False,
                "local_files_only": True,
                "cache_hit": cache_hit,
                "formatted_prompt_prefix": formatted,
            },
            precomputed_logit_rank=scoring.raw_score,
            precomputed_norm_ll=scoring.normalized_log_likelihood,
            scoring_compute_ms=scoring.scoring_compute_ms,
        )

    def score_response(
        self,
        prompt_id: str,
        prompt_text: str,
        response_text: str,
    ) -> ScoringResult:
        if not isinstance(response_text, str) or not response_text.strip():
            raise AdapterError(
                "NO_VALID_RESPONSE_TOKENS",
                "Cannot score an empty or whitespace-only response_text.",
            )

        engine, tokenizer, model, meta, cache_hit = self._load_components()
        formatted_prompt = validate_and_format_chat_prompt(
            prompt_text,
            explicit_override=self.profile.get("chat_template"),
            tokenizer_chat_template=meta.get("chat_template"),
        )

        prompt_ids = list(tokenizer.encode(formatted_prompt, add_special_tokens=True))
        resp_ids = list(tokenizer.encode(response_text, add_special_tokens=False))
        if not resp_ids:
            raise AdapterError(
                "NO_VALID_RESPONSE_TOKENS",
                "Tokenizing response_text produced zero response tokens.",
            )

        full_ids = prompt_ids + resp_ids

        if engine == "offline_sharded_causal_lm":
            logits_np = model.forward_logits(full_ids)
        else:
            torch = meta["torch"]
            device = str(self.profile.get("device") or "cpu")
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_tensor)
                logits_np = outputs.logits[0].detach().cpu().to(torch.float64).numpy()

        res = compute_teacher_forcing_logit_ranks(
            full_input_ids=full_ids,
            prompt_token_len=len(prompt_ids),
            logits_matrix=logits_np,
            pad_token_id=meta.get("pad_token_id"),
        )
        res.meta["cache_hit"] = cache_hit
        res.meta["formatted_prompt_prefix"] = formatted_prompt
        return res
