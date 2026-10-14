"""Tests for LocalTransformersAdapter (sharded checkpoints, tokenizer chat templates, caching, error handling), FastAPI endpoints, and CLI."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from fastapi.testclient import TestClient

from modeltrace.adapters.base import AdapterError
from modeltrace.adapters.local_transformers import (
    LocalTransformersAdapter,
    TinyRandomCausalLMFixture,
    compute_teacher_forcing_logit_ranks,
    create_offline_sharded_causal_lm_checkpoint,
    detect_local_inference_dependencies,
)
from modeltrace.api.app import create_app
from modeltrace.cli import cli
from modeltrace.schemas import AuditVerdict, ExecutionState

HAS_TORCH_AND_TRANSFORMERS, MISSING_HF_PKGS, _ = detect_local_inference_dependencies()


class TestLocalTransformersAndApiCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "test_api_cli.db")
        self.exports_dir = str(Path(self.tmp_dir.name) / "exports")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_offline_tiny_random_weight_causal_lm_teacher_forcing_fixture(self) -> None:
        fixture = TinyRandomCausalLMFixture(vocab_size=128, hidden_dim=16, seed=2026)
        res1 = fixture.score_public_response(
            prompt_text="Extract the total from the invoice.",
            response_text="Total is 1600.00 USD.",
        )
        res2 = fixture.score_public_response(
            prompt_text="Extract the total from the invoice.",
            response_text="Total is 1600.00 USD.",
        )
        self.assertAlmostEqual(res1.raw_score, res2.raw_score, places=9)
        self.assertIsNotNone(res1.normalized_log_likelihood)
        self.assertGreater(res1.token_count, 5)
        self.assertEqual(res1.token_count, len(res1.meta["per_token_log_ranks"]))

        full_ids = [1, 10, 20, 30, 0]
        logits = fixture.forward_logits(full_ids)
        logits[1, :] = -5.0
        logits[1, 20] = 10.0
        logits[2, :] = -5.0
        logits[2, 99] = 12.0
        logits[2, 30] = 8.0

        manual = compute_teacher_forcing_logit_ranks(
            full_input_ids=full_ids,
            prompt_token_len=2,
            logits_matrix=logits,
            pad_token_id=0,
        )
        self.assertEqual(manual.token_count, 2)
        expected_mean_log_rank = (0.0 + math.log(2.0)) / 2.0
        self.assertAlmostEqual(manual.raw_score, expected_mean_log_rank, places=7)

    def test_local_transformers_adapter_sharded_checkpoint_generation_scoring_caching_and_errors(
        self,
    ) -> None:
        ckpt_dir = Path(self.tmp_dir.name) / "sharded_offline_lm"
        ckpt_info = create_offline_sharded_causal_lm_checkpoint(
            str(ckpt_dir),
            vocab_size=128,
            hidden_dim=16,
            seed=4040,
            chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(len(ckpt_info["shards"]), 2)

        LocalTransformersAdapter.clear_cache()
        profile = {
            "id": "prof_local_sharded",
            "name": "Local Sharded CausalLM",
            "adapter_type": "local_transformers",
            "model_identifier": "offline-sharded-causal-lm-v1",
            "checkpoint_path": str(ckpt_dir),
            "checkpoint_revision": "main",
            "tokenizer_path": str(ckpt_dir),
            "chat_template": None,
            "precision": "float32",
            "device": "cpu",
        }
        adapter = LocalTransformersAdapter(profile=profile, allowed_models_dir=self.tmp_dir.name)

        # 1. Verify capability check validates sharded index and shards
        cap = adapter.check_capability()
        self.assertEqual(cap["status"], "available")
        self.assertEqual(cap["weight_layout"], "sharded")
        self.assertEqual(
            cap["shard_files"],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )

        # 2. Run actual generation and scoring through LocalTransformersAdapter and verify caching
        gen1 = adapter.generate(
            prompt_id="p1",
            prompt_text="Summarize the invoice total.",
            temperature=0.7,
            top_p=0.95,
            max_tokens=16,
            seed=12345,
        )
        self.assertEqual(gen1.token_count, 16)
        self.assertTrue(len(gen1.response_text) >= 16)
        self.assertFalse(gen1.observable_meta["cache_hit"])  # First load is a miss, internal score_response is a hit

        sc1 = adapter.score_response("p1", "Summarize the invoice total.", gen1.response_text)
        self.assertTrue(sc1.meta["cache_hit"])
        self.assertIn("<|im_start|>user\nSummarize the invoice total.", sc1.meta["formatted_prompt_prefix"])
        self.assertAlmostEqual(gen1.precomputed_logit_rank, sc1.raw_score, places=9)

        # Run 4 more scoring calls and verify model/tokenizer was loaded only once (misses == 1)
        for _ in range(4):
            _ = adapter.score_response("p1", "Summarize the invoice total.", "Sample response text.")
        stats = LocalTransformersAdapter.get_cache_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hits"], 5)

        # 3. Independently verify causal alignment against direct fixture forward pass
        ref_fixture = TinyRandomCausalLMFixture(
            vocab_size=128,
            hidden_dim=16,
            seed=4040,
            chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        )
        direct_score = ref_fixture.score_public_response(
            prompt_text="Summarize the invoice total.",
            response_text=gen1.response_text,
        )
        self.assertAlmostEqual(sc1.raw_score, direct_score.raw_score, places=9)
        self.assertAlmostEqual(
            sc1.normalized_log_likelihood, direct_score.normalized_log_likelihood, places=9
        )

        # 4. Verify validated explicit chat_template override vs invalid override
        override_prof = dict(profile, chat_template="[INST] {prompt} [/INST]\n")
        override_adapter = LocalTransformersAdapter(
            profile=override_prof, allowed_models_dir=self.tmp_dir.name
        )
        sc_override = override_adapter.score_response(
            "p1", "Summarize the invoice total.", gen1.response_text
        )
        self.assertEqual(
            sc_override.meta["formatted_prompt_prefix"],
            "[INST] Summarize the invoice total. [/INST]\n",
        )
        self.assertNotAlmostEqual(sc_override.raw_score, sc1.raw_score, places=4)

        bad_tpl_prof = dict(profile, chat_template="STATIC_TEMPLATE_WITHOUT_PLACEHOLDER")
        bad_tpl_adapter = LocalTransformersAdapter(
            profile=bad_tpl_prof, allowed_models_dir=self.tmp_dir.name
        )
        self.assertEqual(bad_tpl_adapter.check_capability()["reason_code"], "INVALID_CHAT_TEMPLATE")
        with self.assertRaises(AdapterError) as ctx_tpl:
            bad_tpl_adapter.score_response("p1", "Hello", "World")
        self.assertEqual(ctx_tpl.exception.error_code, "INVALID_CHAT_TEMPLATE")

        # 5. Verify clear failures on empty/no-token and non-finite scoring
        with self.assertRaises(AdapterError) as ctx_empty:
            adapter.score_response("p1", "Hello", "   ")
        self.assertEqual(ctx_empty.exception.error_code, "NO_VALID_RESPONSE_TOKENS")

        with self.assertRaises(AdapterError) as ctx_all_pad:
            compute_teacher_forcing_logit_ranks(
                full_input_ids=[1, 10, 0, 0],
                prompt_token_len=2,
                logits_matrix=np.zeros((4, 16), dtype=np.float64),
                pad_token_id=0,
            )
        self.assertEqual(ctx_all_pad.exception.error_code, "NO_VALID_RESPONSE_TOKENS")

        bad_logits = np.zeros((4, 16), dtype=np.float64)
        bad_logits[1, 5] = float("nan")
        with self.assertRaises(AdapterError) as ctx_nan:
            compute_teacher_forcing_logit_ranks(
                full_input_ids=[1, 10, 5, 6],
                prompt_token_len=2,
                logits_matrix=bad_logits,
                pad_token_id=0,
            )
        self.assertEqual(ctx_nan.exception.error_code, "NON_FINITE_LOGITS")

        # 6. Verify missing shard detection on sharded index
        (ckpt_dir / "model-00002-of-00002.safetensors").unlink()
        cap_missing_shard = adapter.check_capability()
        self.assertEqual(cap_missing_shard["status"], "unavailable")
        self.assertEqual(cap_missing_shard["reason_code"], "MISSING_CHECKPOINT_SHARD")

    def test_local_transformers_unavailable_capability_diagnostics(self) -> None:
        adapter = LocalTransformersAdapter(
            profile={
                "checkpoint_path": "nonexistent_local_checkpoint",
                "model_identifier": "local-llama",
            },
            allowed_models_dir=self.tmp_dir.name,
        )
        cap = adapter.check_capability()
        self.assertEqual(cap["status"], "unavailable")
        self.assertFalse(cap["trust_remote_code"])
        self.assertTrue(cap["local_files_only"])
        self.assertIn(
            cap["reason_code"],
            (
                "MISSING_DEPENDENCIES",
                "MISSING_CHECKPOINT_DIRECTORY",
                "MISSING_CHECKPOINT_WEIGHTS",
            ),
        )

    @unittest.skipUnless(
        HAS_TORCH_AND_TRANSFORMERS,
        f"NOT RUN: Optional HuggingFace packages ({', '.join(MISSING_HF_PKGS)}) are not installed in offline sandbox VM.",
    )
    def test_live_huggingface_tiny_random_checkpoint_if_installed(self) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # type: ignore[import-not-found]
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-not-found]

        ckpt_dir = Path(self.tmp_dir.name) / "tiny_hf_gpt2"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        for i in range(3, 64):
            vocab[f"tok_{i}"] = i
        tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok_3"))
        tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok_obj,
            unk_token="tok_3",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            chat_template="<|user|>\n{prompt}\n<|assistant|>\n",
        )
        fast_tok.save_pretrained(str(ckpt_dir))

        cfg = GPT2Config(
            vocab_size=64, n_positions=64, n_embd=16, n_layer=1, n_head=2, pad_token_id=0, bos_token_id=1, eos_token_id=2
        )
        torch.manual_seed(2026)
        model = GPT2LMHeadModel(cfg)
        model.save_pretrained(str(ckpt_dir))

        LocalTransformersAdapter.clear_cache()
        adapter = LocalTransformersAdapter(
            profile={
                "checkpoint_path": str(ckpt_dir),
                "tokenizer_path": str(ckpt_dir),
                "model_identifier": "tiny-hf-gpt2",
                "precision": "float32",
                "device": "cpu",
            },
            allowed_models_dir=self.tmp_dir.name,
        )
        cap = adapter.check_capability()
        self.assertEqual(cap["status"], "available")
        gen = adapter.generate("p1", "tok_4 tok_5", 0.7, 0.95, 8, seed=42)
        self.assertGreater(gen.token_count, 0)
        sc = adapter.score_response("p1", "tok_4 tok_5", gen.response_text)
        self.assertTrue(math.isfinite(sc.raw_score))

    def test_fastapi_endpoints_security_middleware_and_exports(self) -> None:
        app = create_app(
            db_path=self.db_path, exports_dir=self.exports_dir, start_worker=False
        )
        client = TestClient(app)

        r_health = client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["tagline"], "Know what your API is serving.")

        r_cap = client.get("/api/capabilities")
        self.assertEqual(r_cap.status_code, 200)
        self.assertTrue(r_cap.json()["adapters"]["synthetic_demo"]["available"])
        self.assertTrue(r_cap.json()["security"]["socket_ip_pinning_enforced"])

        # Reject untrusted Host header even if Origin matches that Host
        r_bad_host = client.post(
            "/api/profiles",
            headers={
                "Host": "evil.example.com",
                "Origin": "http://evil.example.com",
                "Content-Type": "application/json",
            },
            json={
                "name": "BlockedHost",
                "adapter_type": "synthetic_demo",
                "model_identifier": "demo",
            },
        )
        self.assertEqual(r_bad_host.status_code, 400)
        self.assertEqual(r_bad_host.json()["error_code"], "DISALLOWED_HOST_HEADER")

        r_bad_origin = client.post(
            "/api/profiles",
            headers={
                "Origin": "https://evil.example.com",
                "Content-Type": "application/json",
            },
            json={
                "name": "Blocked",
                "adapter_type": "synthetic_demo",
                "model_identifier": "demo",
            },
        )
        self.assertEqual(r_bad_origin.status_code, 403)
        self.assertEqual(r_bad_origin.json()["error_code"], "DISALLOWED_ORIGIN")

        r_bad_ct = client.post(
            "/api/profiles",
            headers={"Content-Type": "text/plain"},
            content="name=bad",
        )
        self.assertEqual(r_bad_ct.status_code, 415)
        self.assertEqual(
            r_bad_ct.json()["error_code"], "JSON_CONTENT_TYPE_REQUIRED"
        )

        r_prev = client.post(
            "/api/audits/preview",
            json={
                "reference_profile_id": "prof_demo_reference",
                "target_profile_id": "prof_demo_target_substituted",
                "score_mode": "local_logit_rank",
                "n_prompts": 8,
                "m_reference": 4,
            },
        )
        self.assertEqual(r_prev.status_code, 200)
        self.assertTrue(r_prev.json()["ready_to_start"])
        self.assertEqual(r_prev.json()["budget"]["nominal_total_calls"], 40)

        r_create = client.post(
            "/api/audits",
            json={
                "name": "API Substituted Demo Audit",
                "reference_profile_id": "prof_demo_reference",
                "target_profile_id": "prof_demo_target_substituted",
                "score_mode": "local_logit_rank",
                "n_prompts": 8,
                "m_reference": 4,
                "start_immediately": False,
                "include_secondary_tests": True,
            },
        )
        self.assertEqual(r_create.status_code, 201)
        aid = r_create.json()["id"]
        self.assertEqual(
            r_create.json()["execution_state"], ExecutionState.CREATED.value
        )

        r_start = client.post(f"/api/audits/{aid}/start?synchronous=true")
        self.assertEqual(r_start.status_code, 200)
        self.assertEqual(
            r_start.json()["execution_state"], ExecutionState.COMPLETED.value
        )
        self.assertEqual(r_start.json()["verdict"], AuditVerdict.INCONSISTENT.value)

        r_double = client.post(f"/api/audits/{aid}/start?synchronous=true")
        self.assertEqual(r_double.status_code, 409)

        r_prog = client.get(f"/api/audits/{aid}/progress")
        self.assertEqual(r_prog.status_code, 200)
        self.assertEqual(r_prog.json()["progress_fraction"], 1.0)

        r_ev = client.get(f"/api/audits/{aid}/events")
        self.assertEqual(r_ev.status_code, 200)
        self.assertGreaterEqual(len(r_ev.json()["events"]), 2)

        for fmt in ("json", "csv", "html"):
            r_exp = client.get(f"/api/audits/{aid}/export?format={fmt}")
            self.assertEqual(r_exp.status_code, 200)
            self.assertIn("X-ModelTrace-Integrity-SHA256", r_exp.headers)
            if fmt == "html":
                self.assertIn("DEMO MODE", r_exp.text)
                self.assertIn("Know what your API is serving.", r_exp.text)

    def test_cli_demo_scenarios_genuine_substituted_mixed(self) -> None:
        runner = CliRunner()

        res_sub = runner.invoke(
            cli,
            [
                "demo-audit",
                "--scenario",
                "substituted",
                "--n-prompts",
                "16",
                "--m-reference",
                "8",
                "--db-path",
                self.db_path,
            ],
        )
        self.assertEqual(res_sub.exit_code, 0, res_sub.output)
        data_sub = json.loads(res_sub.output)
        self.assertTrue(data_sub["is_demo"])
        self.assertEqual(data_sub["verdict"], AuditVerdict.INCONSISTENT.value)

        res_mix = runner.invoke(
            cli,
            [
                "demo-audit",
                "--scenario",
                "mixed_routing",
                "--n-prompts",
                "20",
                "--m-reference",
                "8",
                "--db-path",
                self.db_path,
            ],
        )
        self.assertEqual(res_mix.exit_code, 0, res_mix.output)
        data_mix = json.loads(res_mix.output)
        self.assertEqual(data_mix["verdict"], AuditVerdict.INCONSISTENT.value)

        res_gen_uncal = runner.invoke(
            cli,
            [
                "demo-audit",
                "--scenario",
                "genuine",
                "--n-prompts",
                "20",
                "--m-reference",
                "8",
                "--no-calibration",
                "--db-path",
                self.db_path,
            ],
        )
        self.assertEqual(res_gen_uncal.exit_code, 0, res_gen_uncal.output)
        data_gen_uncal = json.loads(res_gen_uncal.output)
        self.assertEqual(data_gen_uncal["verdict"], AuditVerdict.INCONCLUSIVE.value)

        res_gen_cal = runner.invoke(
            cli,
            [
                "demo-audit",
                "--scenario",
                "genuine",
                "--n-prompts",
                "20",
                "--m-reference",
                "8",
                "--with-calibration",
                "--export-format",
                "html",
                "--db-path",
                self.db_path,
            ],
        )
        self.assertEqual(res_gen_cal.exit_code, 0, res_gen_cal.output)
        data_gen_cal = json.loads(res_gen_cal.output)
        self.assertEqual(
            data_gen_cal["verdict"],
            AuditVerdict.CONSISTENT_WITHIN_TESTED_LIMITS.value,
        )
