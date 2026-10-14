"""Adapter A: Seeded stochastic synthetic reference and target adapter (DEMO mode).

Supports three realistic stochastic scenarios without external API keys, GPUs, or downloads:
1. `genuine`: Target samples from the exact same prompt-conditional token logit distribution as reference.
2. `substituted`: Target samples from a perturbed/quantized substitute distribution with systematic rank shift and altered response lengths.
3. `mixed_routing`: Target routes each query stochastically (with probability `mixed_routing_bad_fraction`, e.g., 0.45) to the substituted model and otherwise to the genuine model.

All outputs are generated dynamically from seeded RNG streams and evaluated by the real teacher-forcing rank scorer over a synthetic vocabulary V=256. Never hardcodes p-values or verdicts.
"""

import hashlib
import time
from typing import Any, Dict, List

import numpy as np

from modeltrace.adapters.base import GenerationResult, ScoringResult
from modeltrace.schemas import ScoreMode, SyntheticScenario
from modeltrace.services.stats import compute_unicode_length_score

SYNTHETIC_VOCAB_SIZE = 256
SYNTHETIC_WORDS: List[str] = [
    f"tok{i:03d}" if i >= 32 else w
    for i, w in enumerate(
        [
            "The", "computed", "result", "indicates", "value", "is", "verified", "under",
            "reference", "constraints", "with", "residual", "index", "at", "node", "alpha",
            "beta", "gamma", "delta", "epsilon", "vector", "matrix", "norm", "bound",
            "valid", "output", "sequence", "complete", "status", "ok", "true", "false",
        ]
        + [f"tok{j:03d}" for j in range(32, SYNTHETIC_VOCAB_SIZE)]
    )
]


def _prompt_reference_logits(prompt_id: str, prompt_text: str, max_positions: int = 64) -> np.ndarray:
    """Construct a deterministic prompt-specific reference logit table of shape (max_positions, V)."""
    digest = hashlib.sha256(f"{prompt_id}::{prompt_text}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed_int)
    # Sharp peaked logits so reference tokens concentrate in top ranks (low log-rank)
    base_logits = rng.normal(loc=0.0, scale=2.4, size=(max_positions, SYNTHETIC_VOCAB_SIZE))
    return base_logits.astype(np.float64)


def _softmax(logits_1d: np.ndarray, temperature: float = 0.7) -> np.ndarray:
    temp = max(float(temperature), 0.05)
    scaled = logits_1d / temp
    shifted = scaled - np.max(scaled)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v)


def score_tokens_against_reference_logits(
    token_ids: List[int],
    ref_logits_matrix: np.ndarray,
) -> ScoringResult:
    """Compute exact teacher-forcing mean log vocabulary rank and mean log-likelihood."""
    t0 = time.perf_counter()
    if not token_ids:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return ScoringResult(
            raw_score=float(np.log(SYNTHETIC_VOCAB_SIZE)),
            normalized_log_likelihood=-float(np.log(SYNTHETIC_VOCAB_SIZE)),
            token_count=0,
            scoring_compute_ms=elapsed,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            meta={"is_demo": True, "vocab_size": SYNTHETIC_VOCAB_SIZE},
        )

    log_ranks: List[float] = []
    log_probs: List[float] = []
    max_pos = ref_logits_matrix.shape[0]

    for pos, tok_id in enumerate(token_ids):
        row = ref_logits_matrix[pos % max_pos]
        tok_logit = float(row[tok_id])
        # 1-based rank: 1 + count of vocabulary items with strictly higher logit
        rank = 1 + int(np.sum(row > tok_logit))
        log_ranks.append(float(np.log(rank)))

        shifted = row - np.max(row)
        log_z = float(np.log(np.sum(np.exp(shifted))))
        log_probs.append(float(shifted[tok_id] - log_z))

    elapsed = (time.perf_counter() - t0) * 1000.0
    return ScoringResult(
        raw_score=float(np.mean(log_ranks)),
        normalized_log_likelihood=float(np.mean(log_probs)),
        token_count=len(token_ids),
        scoring_compute_ms=elapsed,
        score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
        meta={"is_demo": True, "vocab_size": SYNTHETIC_VOCAB_SIZE},
    )


class SyntheticDemoAdapter:
    """Seeded stochastic DEMO adapter implementing generation, scoring, and capability inspection."""

    def __init__(self, profile: Dict[str, Any]) -> None:
        self.profile = profile
        self.scenario = str(
            profile.get("synthetic_scenario") or SyntheticScenario.GENUINE.value
        ).lower()
        self.bad_fraction = float(profile.get("mixed_routing_bad_fraction", 0.45))

    def check_capability(self) -> Dict[str, Any]:
        return {
            "status": "available",
            "adapter_type": "synthetic_demo",
            "is_demo": True,
            "scenario": self.scenario,
            "mixed_routing_bad_fraction": self.bad_fraction,
            "supports_generation": True,
            "supports_local_logit_scoring": True,
            "vocab_size": SYNTHETIC_VOCAB_SIZE,
            "note": "DEMO synthetic stochastic adapter. No external API calls or model weights required.",
        }

    def generate(
        self,
        prompt_id: str,
        prompt_text: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        rng: np.random.Generator,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        ref_logits = _prompt_reference_logits(prompt_id, prompt_text, max_positions=max(32, max_tokens))

        # Determine whether this specific call routes to genuine or substituted behavior
        routed_to_substitute = False
        if self.scenario == SyntheticScenario.SUBSTITUTED.value:
            routed_to_substitute = True
        elif self.scenario == SyntheticScenario.MIXED_ROUTING.value:
            routed_to_substitute = bool(rng.random() < self.bad_fraction)

        # Sample response token length from scenario-specific length distribution
        cap = min(int(max_tokens), 128)
        if not routed_to_substitute:
            # Genuine: length centered around 18..26 tokens
            base_len = int(rng.integers(18, 27))
        else:
            # Substituted: noticeably longer/verbose or shifted length (34..46 tokens)
            base_len = int(rng.integers(34, 47))
        seq_len = max(4, min(cap, base_len))

        token_ids: List[int] = []
        for pos in range(seq_len):
            row = ref_logits[pos % ref_logits.shape[0]].copy()
            if routed_to_substitute:
                # Perturb logits deterministically + add quantization noise to simulate substituted model
                shift_vec = np.sin(np.arange(SYNTHETIC_VOCAB_SIZE, dtype=np.float64) * (pos + 1.7)) * 3.2
                row = -0.35 * row + shift_vec
            probs = _softmax(row, temperature=temperature)
            tok = int(rng.choice(SYNTHETIC_VOCAB_SIZE, p=probs))
            token_ids.append(tok)

        words = [SYNTHETIC_WORDS[t] for t in token_ids]
        if routed_to_substitute:
            # Also make each token string slightly padded so Unicode length reflects substitution
            response_text = "[DEMO-SUB] " + " ".join(f"{w}_alt" for w in words)
        else:
            response_text = "[DEMO] " + " ".join(words)

        scoring = score_tokens_against_reference_logits(token_ids, ref_logits)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return GenerationResult(
            response_text=response_text,
            token_count=len(token_ids),
            finish_reason="stop",
            declared_model=str(self.profile.get("model_identifier", "demo-model")),
            latency_ms=elapsed_ms,
            http_status=200,
            retry_count=0,
            observable_meta={
                "is_demo": True,
                "configured_scenario": self.scenario,
                "routed_to_substitute": routed_to_substitute,
                "token_ids": token_ids,
                "declared_model_untrusted": True,
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
        observable_meta: Dict[str, Any],
        score_mode: str,
    ) -> ScoringResult:
        if score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value:
            t0 = time.perf_counter()
            u_len = compute_unicode_length_score(response_text)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ScoringResult(
                raw_score=u_len,
                normalized_log_likelihood=None,
                token_count=len((response_text or "").split()),
                scoring_compute_ms=elapsed,
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY.value,
                meta={"is_demo": True, "scope": "LENGTH_DISTRIBUTION_ONLY"},
            )

        # Retokenize public text back into SYNTHETIC_WORDS IDs for consistent public-text retokenization
        word_to_id = {w: idx for idx, w in enumerate(SYNTHETIC_WORDS)}
        cleaned_tokens: List[int] = []
        for raw_tok in (response_text or "").split():
            if raw_tok in ("[DEMO]", "[DEMO-SUB]"):
                continue
            base_tok = raw_tok[:-4] if raw_tok.endswith("_alt") else raw_tok
            if base_tok in word_to_id:
                cleaned_tokens.append(word_to_id[base_tok])
            else:
                # Hash unknown public token into synthetic vocab deterministically
                h = int(hashlib.md5(raw_tok.encode("utf-8")).hexdigest()[:6], 16)
                cleaned_tokens.append(h % SYNTHETIC_VOCAB_SIZE)

        ref_logits = _prompt_reference_logits(
            prompt_id, prompt_text, max_positions=max(64, len(cleaned_tokens) + 4)
        )
        return score_tokens_against_reference_logits(cleaned_tokens, ref_logits)
