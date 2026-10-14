"""Tests for real OpenAI-compatible HTTP adapter against a controlled local server, worker crash/resume/caps/cancellation, and frozen profile/scorer snapshots."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from modeltrace.adapters.base import AdapterError
from modeltrace.adapters.openai_compatible import OpenAICompatibleAdapter
from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AdapterType,
    AttemptStatus,
    AuditCreateRequest,
    AuditPreviewRequest,
    AuditVerdict,
    ExecutionState,
    ProfileCreate,
    ProfileUpdate,
    ScoreMode,
    SettingsUpdateRequest,
)
from modeltrace.services.worker import AuditWorkerService


class _ControlledMockOpenAIHandler(BaseHTTPRequestHandler):
    """Programmable HTTP handler simulating OpenAI-compatible chat completions and failure modes."""

    mode: str = "ok"
    call_counter: int = 0
    received_headers: List[Dict[str, str]] = []
    received_models: List[str] = []

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        type(self).call_counter += 1
        cur_call = type(self).call_counter
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        type(self).received_headers.append(
            {"Authorization": self.headers.get("Authorization", "")}
        )

        mode = type(self).mode

        if mode == "auth_401":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            auth_val = self.headers.get("Authorization", "")
            self.wfile.write(
                json.dumps({"error": f"Invalid key in {auth_val}"}).encode("utf-8")
            )
            return

        if mode == "flaky_429_once" and cur_call == 1:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Rate limit exceeded, retry shortly"}')
            return

        if mode == "always_429":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Hard rate limit 429"}')
            return

        if mode == "flaky_500_once" and cur_call == 1:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Upstream gateway error 502"}')
            return

        if mode == "always_500":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Internal server error 500"}')
            return

        if mode == "malformed_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": "not_an_object"}]}')
            return

        if mode == "redirect_302":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/redirected")
            self.end_headers()
            return

        if mode == "slow_timeout":
            time.sleep(1.2)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        body_obj = json.loads(raw) if raw else {}
        req_model = str(body_obj.get("model", "mock-model"))
        type(self).received_models.append(req_model)
        content = f"Verified mock completion #{cur_call} for model {req_model}."
        resp_payload = {
            "id": f"chatcmpl-{cur_call}",
            "object": "chat.completion",
            "model": f"{req_model}-declared-header",
            "system_fingerprint": "fp_local_test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_payload).encode("utf-8"))


class TestHttpAdapterAndWorkerLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ControlledMockOpenAIHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}/v1"
        cls.origin = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "test_worker.db")
        os.environ["MODELTRACE_PROVIDER_MOCK_SECRET"] = "secret_live_key_abc987654"
        _ControlledMockOpenAIHandler.mode = "ok"
        _ControlledMockOpenAIHandler.call_counter = 0
        _ControlledMockOpenAIHandler.received_headers.clear()
        _ControlledMockOpenAIHandler.received_models.clear()

    def tearDown(self) -> None:
        os.environ.pop("MODELTRACE_PROVIDER_MOCK_SECRET", None)
        self.tmp_dir.cleanup()

    def _make_adapter(
        self, timeout_seconds: float = 2.0, max_retries: int = 2
    ) -> OpenAICompatibleAdapter:
        prof = {
            "id": "prof_http_test",
            "name": "Local HTTP Test",
            "adapter_type": "openai_compatible",
            "base_url": self.base_url,
            "api_key_env_var": "MODELTRACE_PROVIDER_MOCK_SECRET",
            "model_identifier": "llama-3-8b-test",
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "extra_config": {},
        }
        return OpenAICompatibleAdapter(
            profile=prof,
            allowed_outbound_origins=[],
            allowed_local_inference_origins=[self.origin],
        )

    def test_http_adapter_ok_flaky429_500_auth401_timeout_malformed_redirect(self) -> None:
        adapter = self._make_adapter(timeout_seconds=0.6, max_retries=2)

        _ControlledMockOpenAIHandler.mode = "ok"
        res = adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertIn("Verified mock completion", res.response_text)
        self.assertEqual(res.retry_count, 0)
        self.assertTrue(res.observable_meta["declared_model_untrusted"])
        self.assertEqual(adapter.last_connected_peer_ips, ["127.0.0.1"])

        _ControlledMockOpenAIHandler.mode = "flaky_429_once"
        _ControlledMockOpenAIHandler.call_counter = 0
        res_retry = adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(res_retry.retry_count, 1)
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 2)

        _ControlledMockOpenAIHandler.mode = "always_429"
        _ControlledMockOpenAIHandler.call_counter = 0
        with self.assertRaises(AdapterError) as ctx:
            adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "RATE_LIMIT_429")
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 3)

        _ControlledMockOpenAIHandler.mode = "flaky_500_once"
        _ControlledMockOpenAIHandler.call_counter = 0
        res_502 = adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(res_502.retry_count, 1)
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 2)

        _ControlledMockOpenAIHandler.mode = "always_500"
        _ControlledMockOpenAIHandler.call_counter = 0
        with self.assertRaises(AdapterError) as ctx:
            adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "SERVER_ERROR")
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 3)

        _ControlledMockOpenAIHandler.mode = "auth_401"
        _ControlledMockOpenAIHandler.call_counter = 0
        with self.assertRaises(AdapterError) as ctx:
            adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "AUTH_ERROR")
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 1)
        self.assertNotIn("secret_live_key_abc987654", ctx.exception.message)
        self.assertIn("[REDACTED_SECRET]", ctx.exception.message)

        _ControlledMockOpenAIHandler.mode = "malformed_json"
        with self.assertRaises(AdapterError) as ctx:
            adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "MALFORMED_RESPONSE")

        _ControlledMockOpenAIHandler.mode = "redirect_302"
        with self.assertRaises(AdapterError) as ctx:
            adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "REDIRECT_FORBIDDEN")

        _ControlledMockOpenAIHandler.mode = "slow_timeout"
        fast_adapter = self._make_adapter(timeout_seconds=0.5, max_retries=0)
        with self.assertRaises(AdapterError) as ctx:
            fast_adapter.generate("p1", "Hello", 0.7, 0.95, 32)
        self.assertEqual(ctx.exception.error_code, "TIMEOUT")

    def test_worker_crash_restart_interruption_and_conservative_resume(self) -> None:
        repo = Repository(self.db_path)
        repo.update_settings(
            SettingsUpdateRequest(allowed_local_inference_origins=[self.origin])
        )
        worker = AuditWorkerService(repo=repo, start_background_loop=False)

        ref_p = repo.create_profile(
            ProfileCreate(
                name="HTTP Ref",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                api_key_env_var="MODELTRACE_PROVIDER_MOCK_SECRET",
                model_identifier="ref-model",
            )
        )
        tgt_p = repo.create_profile(
            ProfileCreate(
                name="HTTP Tgt",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                api_key_env_var="MODELTRACE_PROVIDER_MOCK_SECRET",
                model_identifier="tgt-model",
            )
        )

        audit = worker.create_audit(
            AuditCreateRequest(
                name="Crash Recovery Test Audit",
                reference_profile_id=ref_p["id"],
                target_profile_id=tgt_p["id"],
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY,
                n_prompts=2,
                m_reference=2,
                max_total_attempts=10,
                start_immediately=False,
            )
        )
        aid = audit["id"]

        conn = repo._conn()
        with conn:
            conn.execute(
                "UPDATE audits SET execution_state = ? WHERE id = ?;",
                (ExecutionState.RUNNING.value, aid),
            )
        conn.close()

        probe_rev = repo.get_default_probe_revision()
        first_pid = probe_rev["prompts"][0]["prompt_id"]
        att1 = repo.create_sample_attempt(aid, first_pid, 0, "reference", 0, 1, ref_p["id"])
        repo.complete_sample_attempt(
            attempt_id=att1,
            audit_id=aid,
            prompt_id=first_pid,
            role="reference",
            sample_index=0,
            response_text="Pre-crash completed response",
            token_count=4,
            unicode_length=28,
            finish_reason="stop",
            declared_model="ref-model",
            observable_meta={},
            raw_score=28.0,
            normalized_log_likelihood=None,
            scoring_compute_ms=0.1,
            latency_ms=5.0,
        )
        _att2_inflight = repo.create_sample_attempt(
            aid, first_pid, 0, "reference", 1, 2, ref_p["id"]
        )

        worker_after_restart = AuditWorkerService(repo=repo, start_background_loop=False)
        interrupted_audit = repo.get_audit(aid)
        assert interrupted_audit is not None
        self.assertEqual(
            interrupted_audit["execution_state"], ExecutionState.INTERRUPTED.value
        )
        self.assertEqual(interrupted_audit["verdict"], AuditVerdict.INCONCLUSIVE.value)
        self.assertEqual(interrupted_audit["uncertain_prior_attempts"], 1)

        with self.assertRaises(RuntimeError):
            worker_after_restart.start_audit(aid, run_synchronously=True)

        _ControlledMockOpenAIHandler.call_counter = 0
        resumed = worker_after_restart.resume_audit(aid, run_synchronously=True)
        self.assertEqual(resumed["execution_state"], ExecutionState.COMPLETED.value)
        self.assertEqual(resumed["resume_count"], 1)
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 5)

        counts = repo.get_attempt_counts(aid)
        self.assertEqual(counts["completed_reference_samples"], 4)
        self.assertEqual(counts["completed_target_samples"], 2)
        self.assertEqual(counts["uncertain_prior_attempts"], 1)
        self.assertEqual(counts["total_attempts_used"], 7)

    def test_frozen_profile_and_scorer_snapshots_across_edit_start_and_interrupt_resume(self) -> None:
        repo = Repository(self.db_path)
        repo.update_settings(
            SettingsUpdateRequest(allowed_local_inference_origins=[self.origin])
        )
        worker = AuditWorkerService(repo=repo, start_background_loop=False)

        ref_p = repo.create_profile(
            ProfileCreate(
                name="Immutable Ref",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                api_key_env_var="MODELTRACE_PROVIDER_MOCK_SECRET",
                model_identifier="frozen-ref-model-v1",
            )
        )
        tgt_p = repo.create_profile(
            ProfileCreate(
                name="Immutable Tgt",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                api_key_env_var="MODELTRACE_PROVIDER_MOCK_SECRET",
                model_identifier="frozen-tgt-model-v1",
            )
        )

        # Case 1: Create -> Edit Profile -> Start
        aud1 = worker.create_audit(
            AuditCreateRequest(
                name="Create Then Edit Before Start",
                reference_profile_id=ref_p["id"],
                target_profile_id=tgt_p["id"],
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY,
                n_prompts=1,
                m_reference=2,
                start_immediately=False,
            )
        )
        # Verify manifest stores env var name, never the resolved secret value
        manifest_raw = json.dumps(aud1["manifest"])
        self.assertIn("MODELTRACE_PROVIDER_MOCK_SECRET", manifest_raw)
        self.assertNotIn("secret_live_key_abc987654", manifest_raw)

        # Mutate both profiles in SQLite BEFORE starting aud1
        repo.update_profile(
            ref_p["id"],
            ProfileUpdate(
                model_identifier="mutated-ref-v2",
                base_url="http://127.0.0.1:9/broken",
            ),
        )
        repo.update_profile(
            tgt_p["id"],
            ProfileUpdate(
                model_identifier="mutated-tgt-v2",
                base_url="http://127.0.0.1:9/broken",
            ),
        )

        _ControlledMockOpenAIHandler.received_models.clear()
        completed1 = worker.start_audit(aud1["id"], run_synchronously=True)
        self.assertEqual(completed1["execution_state"], ExecutionState.COMPLETED.value)
        self.assertEqual(
            _ControlledMockOpenAIHandler.received_models,
            ["frozen-ref-model-v1", "frozen-ref-model-v1", "frozen-tgt-model-v1"],
        )

        # Case 2: Interrupt -> Edit Profile -> Resume
        # Restore valid base_url temporarily to create aud2 with frozen v1-interrupt identifiers
        repo.update_profile(
            ref_p["id"],
            ProfileUpdate(
                model_identifier="frozen-ref-interrupt-v1",
                base_url=self.base_url,
            ),
        )
        repo.update_profile(
            tgt_p["id"],
            ProfileUpdate(
                model_identifier="frozen-tgt-interrupt-v1",
                base_url=self.base_url,
            ),
        )
        aud2 = worker.create_audit(
            AuditCreateRequest(
                name="Interrupt Then Edit Before Resume",
                reference_profile_id=ref_p["id"],
                target_profile_id=tgt_p["id"],
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY,
                n_prompts=1,
                m_reference=2,
                start_immediately=False,
            )
        )
        aid2 = aud2["id"]
        conn = repo._conn()
        with conn:
            conn.execute(
                "UPDATE audits SET execution_state = ? WHERE id = ?;",
                (ExecutionState.RUNNING.value, aid2),
            )
        conn.close()
        worker_restart = AuditWorkerService(repo=repo, start_background_loop=False)
        self.assertEqual(
            repo.get_audit(aid2)["execution_state"], ExecutionState.INTERRUPTED.value
        )

        # Mutate profiles after interruption BEFORE calling resume_audit
        repo.update_profile(
            ref_p["id"],
            ProfileUpdate(
                model_identifier="mutated-after-interrupt-v3",
                base_url="http://127.0.0.1:9/broken",
            ),
        )
        repo.update_profile(
            tgt_p["id"],
            ProfileUpdate(
                model_identifier="mutated-after-interrupt-v3",
                base_url="http://127.0.0.1:9/broken",
            ),
        )

        _ControlledMockOpenAIHandler.received_models.clear()
        resumed2 = worker_restart.resume_audit(aid2, run_synchronously=True)
        self.assertEqual(resumed2["execution_state"], ExecutionState.COMPLETED.value)
        self.assertEqual(
            _ControlledMockOpenAIHandler.received_models,
            [
                "frozen-ref-interrupt-v1",
                "frozen-ref-interrupt-v1",
                "frozen-tgt-interrupt-v1",
            ],
        )

    def test_attempt_cap_enforcement_on_retries(self) -> None:
        repo = Repository(self.db_path)
        repo.update_settings(
            SettingsUpdateRequest(allowed_local_inference_origins=[self.origin])
        )
        worker = AuditWorkerService(repo=repo, start_background_loop=False)

        flaky_prof = repo.create_profile(
            ProfileCreate(
                name="Always 429 Target",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                api_key_env_var="MODELTRACE_PROVIDER_MOCK_SECRET",
                model_identifier="flaky-model",
                max_retries=3,
            )
        )
        audit = worker.create_audit(
            AuditCreateRequest(
                name="Attempt Cap Test",
                reference_profile_id=flaky_prof["id"],
                target_profile_id=flaky_prof["id"],
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY,
                n_prompts=1,
                m_reference=1,
                max_reference_attempts=2,
                max_target_attempts=2,
                max_total_attempts=2,
            )
        )
        _ControlledMockOpenAIHandler.mode = "always_429"
        _ControlledMockOpenAIHandler.call_counter = 0
        finished = worker.start_audit(audit["id"], run_synchronously=True)
        self.assertEqual(finished["execution_state"], ExecutionState.FAILED.value)
        self.assertEqual(finished["verdict"], AuditVerdict.INCONCLUSIVE.value)
        self.assertEqual(_ControlledMockOpenAIHandler.call_counter, 2)

    def test_worker_cooperative_cancellation_and_missing_configs(self) -> None:
        repo = Repository(self.db_path)
        worker = AuditWorkerService(repo=repo, start_background_loop=False)

        with self.assertRaises(ValueError):
            worker.preview_audit(
                AuditPreviewRequest(
                    reference_profile_id="prof_nonexistent",
                    target_profile_id="prof_demo_target_genuine",
                )
            )

        http_prof = repo.create_profile(
            ProfileCreate(
                name="Live HTTP Profile",
                adapter_type=AdapterType.OPENAI_COMPATIBLE,
                base_url=self.base_url,
                model_identifier="live-model",
            )
        )
        with self.assertRaises(ValueError) as ctx:
            worker.preview_audit(
                AuditPreviewRequest(
                    reference_profile_id="prof_demo_reference",
                    target_profile_id=http_prof["id"],
                )
            )
        self.assertIn("Cannot mix DEMO", str(ctx.exception))

        audit = worker.create_audit(
            AuditCreateRequest(
                name="Cancelable Audit",
                reference_profile_id="prof_demo_reference",
                target_profile_id="prof_demo_target_genuine",
                score_mode=ScoreMode.LOCAL_LOGIT_RANK,
                n_prompts=4,
                m_reference=2,
                start_immediately=False,
            )
        )
        cancelled = worker.cancel_audit(audit["id"])
        self.assertEqual(cancelled["execution_state"], ExecutionState.CANCELLED.value)
        self.assertEqual(cancelled["verdict"], AuditVerdict.INCONCLUSIVE.value)
