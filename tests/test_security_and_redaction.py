"""Tests for secret env enforcement, redaction across logs/exports, CSV formula neutralization, hostile HTML escaping, socket-level IP pinning, DNS rebinding protection, and inbound Host/Origin validation."""

import csv
import io
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

from modeltrace.adapters.base import AdapterError
from modeltrace.adapters.openai_compatible import OpenAICompatibleAdapter
from modeltrace.db.repository import Repository
from modeltrace.schemas import AuditCreateRequest, ScoreMode
from modeltrace.services.exporter import ExportService, neutralize_csv_cell
from modeltrace.services.worker import AuditWorkerService
from modeltrace.utils.security import (
    PinnedIPSyncBackend,
    SecurityViolationError,
    redact_secrets,
    redact_text,
    resolve_secret_value,
    validate_inbound_localhost_request,
    validate_local_checkpoint_path,
    validate_outbound_url,
    validate_secret_env_var_name,
)


class TestSecurityAndRedaction(unittest.TestCase):
    def test_secret_env_var_validation_and_redaction(self) -> None:
        self.assertEqual(
            validate_secret_env_var_name("MODELTRACE_PROVIDER_CUSTOM_KEY"),
            "MODELTRACE_PROVIDER_CUSTOM_KEY",
        )
        for bad_name in ["sk-live-1234567890abcdef", "OPENAI_API_KEY", "PATH", "modeltrace_provider_low"]:
            with self.assertRaises(SecurityViolationError):
                validate_secret_env_var_name(bad_name)

        os.environ["MODELTRACE_PROVIDER_UNITTEST_KEY"] = "super_secret_token_val_998877"
        try:
            resolved = resolve_secret_value("MODELTRACE_PROVIDER_UNITTEST_KEY")
            self.assertEqual(resolved, "super_secret_token_val_998877")

            leaked = (
                "Error calling API with key super_secret_token_val_998877 and "
                "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 "
                "and fallback sk-proj1234567890abcdef"
            )
            cleaned = redact_text(leaked)
            self.assertNotIn("super_secret_token_val_998877", cleaned)
            self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", cleaned)
            self.assertNotIn("sk-proj1234567890abcdef", cleaned)
            self.assertIn("[REDACTED_SECRET]", cleaned)

            nested = {
                "authorization": "Bearer secret123456",
                "nested_msg": "Token super_secret_token_val_998877 failed",
            }
            clean_dict = redact_secrets(nested)
            self.assertEqual(clean_dict["authorization"], "[REDACTED_SECRET]")
            self.assertNotIn("super_secret_token_val_998877", clean_dict["nested_msg"])
        finally:
            os.environ.pop("MODELTRACE_PROVIDER_UNITTEST_KEY", None)

    def test_outbound_url_and_ip_guards(self) -> None:
        # 1. Reject credentials in URL
        with self.assertRaises(SecurityViolationError) as ctx:
            validate_outbound_url(
                "http://admin:secret@127.0.0.1:8080/v1/chat/completions",
                allowed_outbound_origins=[],
                allowed_local_inference_origins=["http://127.0.0.1:8080"],
            )
        self.assertEqual(ctx.exception.code, "CREDENTIALS_IN_URL_FORBIDDEN")

        # 2. Reject non-http(s) schemes
        with self.assertRaises(SecurityViolationError) as ctx:
            validate_outbound_url(
                "file:///etc/passwd",
                allowed_outbound_origins=[],
                allowed_local_inference_origins=[],
            )
        self.assertEqual(ctx.exception.code, "INVALID_URL_SCHEME")

        # 3. Reject link-local/metadata IP via mocked resolver without touching external network
        def _mock_metadata_resolver(host: str, port: int) -> List[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

        with self.assertRaises(SecurityViolationError) as ctx:
            validate_outbound_url(
                "http://local-meta-fixture.invalid:8080/v1/chat/completions",
                allowed_outbound_origins=[],
                allowed_local_inference_origins=["http://local-meta-fixture.invalid:8080"],
                resolver_fn=_mock_metadata_resolver,
            )
        self.assertEqual(ctx.exception.code, "METADATA_DESTINATION_FORBIDDEN")

        # 4. Reject loopback/private IP when only in public outbound allowlist (not in local allowlist)
        with self.assertRaises(SecurityViolationError) as ctx:
            validate_outbound_url(
                "https://127.0.0.1:9001/v1/chat/completions",
                allowed_outbound_origins=["https://127.0.0.1:9001"],
                allowed_local_inference_origins=[],
            )
        self.assertEqual(ctx.exception.code, "PRIVATE_DESTINATION_FORBIDDEN")

        # 5. Reject http:// for public outbound origin (requires https://)
        with self.assertRaises(SecurityViolationError) as ctx:
            validate_outbound_url(
                "http://public-api.example.invalid:80/v1/chat/completions",
                allowed_outbound_origins=["http://public-api.example.invalid:80"],
                allowed_local_inference_origins=[],
                resolve_dns=False,
            )
        self.assertEqual(ctx.exception.code, "HTTPS_REQUIRED_FOR_PUBLIC_ORIGIN")

        # 6. Permit loopback IP when exact origin is in allowed_local_inference_origins
        url_ok, origin_ok, ips_ok = validate_outbound_url(
            "http://127.0.0.1:9001/v1/chat/completions",
            allowed_outbound_origins=[],
            allowed_local_inference_origins=["http://127.0.0.1:9001"],
        )
        self.assertEqual(origin_ok, "http://127.0.0.1:9001")
        self.assertIn("127.0.0.1", ips_ok)

        # 7. Reject local checkpoint path traversal
        with self.assertRaises(SecurityViolationError) as ctx:
            validate_local_checkpoint_path("../../etc/shadow", "/tmp/models")
        self.assertEqual(ctx.exception.code, "CHECKPOINT_PATH_TRAVERSAL")

    def test_socket_ip_pinning_dns_rebinding_and_inbound_host_origin_guards(self) -> None:
        # 1a. Simulated TOCTOU DNS rebinding to a different public IP (93.184.216.34 -> 93.184.216.99)
        resolution_calls = {"count": 0}

        def _rebinding_public_resolver(host: str, port: int) -> List[Any]:
            resolution_calls["count"] += 1
            if resolution_calls["count"] == 1:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.99", port))]

        os.environ["MODELTRACE_PROVIDER_REBIND_TEST_KEY"] = "secret_never_sent_on_rebind"
        try:
            adapter = OpenAICompatibleAdapter(
                profile={
                    "id": "prof_rebind",
                    "name": "Rebind Test",
                    "adapter_type": "openai_compatible",
                    "base_url": "https://rebind-fixture.invalid:8443/v1",
                    "api_key_env_var": "MODELTRACE_PROVIDER_REBIND_TEST_KEY",
                    "model_identifier": "test-model",
                    "timeout_seconds": 1.0,
                    "max_retries": 0,
                },
                allowed_outbound_origins=["https://rebind-fixture.invalid:8443"],
                allowed_local_inference_origins=[],
                resolver_fn=_rebinding_public_resolver,
            )
            with self.assertRaises(AdapterError) as ctx:
                adapter.generate("p1", "Hello", 0.7, 0.95, 16)
            self.assertEqual(ctx.exception.error_code, "DNS_REBINDING_DETECTED")
            self.assertNotIn("secret_never_sent_on_rebind", ctx.exception.message)
            self.assertEqual(adapter.last_connected_peer_ips, [])

            # 1b. Simulated TOCTOU DNS rebinding to a private RFC1918 IP (93.184.216.34 -> 10.99.0.1)
            priv_calls = {"count": 0}

            def _rebinding_private_resolver(host: str, port: int) -> List[Any]:
                priv_calls["count"] += 1
                if priv_calls["count"] == 1:
                    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.99.0.1", port))]

            adapter_priv = OpenAICompatibleAdapter(
                profile={
                    "id": "prof_rebind_priv",
                    "name": "Rebind Private Test",
                    "adapter_type": "openai_compatible",
                    "base_url": "https://rebind-fixture.invalid:8443/v1",
                    "api_key_env_var": "MODELTRACE_PROVIDER_REBIND_TEST_KEY",
                    "model_identifier": "test-model",
                    "timeout_seconds": 1.0,
                    "max_retries": 0,
                },
                allowed_outbound_origins=["https://rebind-fixture.invalid:8443"],
                allowed_local_inference_origins=[],
                resolver_fn=_rebinding_private_resolver,
            )
            with self.assertRaises(AdapterError) as ctx_priv:
                adapter_priv.generate("p1", "Hello", 0.7, 0.95, 16)
            self.assertEqual(ctx_priv.exception.error_code, "PRIVATE_DESTINATION_FORBIDDEN")
            self.assertNotIn("secret_never_sent_on_rebind", ctx_priv.exception.message)
        finally:
            os.environ.pop("MODELTRACE_PROVIDER_REBIND_TEST_KEY", None)

        # 2. PinnedIPSyncBackend rejects unexpected connect host or port
        backend = PinnedIPSyncBackend(
            expected_host="approved.local",
            expected_port=8080,
            pinned_ips=["127.0.0.1"],
            allow_local_inference=True,
            url_for_recheck="http://approved.local:8080/v1",
        )
        with self.assertRaises(SecurityViolationError) as ctx2:
            backend.connect_tcp("unapproved.local", 8080)
        self.assertEqual(ctx2.exception.code, "UNEXPECTED_CONNECT_TARGET")

        # 3. Inbound Host and Origin validation: never trust Host to define same-origin
        with self.assertRaises(SecurityViolationError) as ctx3:
            validate_inbound_localhost_request(
                host_header="evil.example.com:8000",
                origin_header="http://evil.example.com:8000",
                referer_header=None,
                method="POST",
                allowed_frontend_origins=["http://127.0.0.1:8000"],
            )
        self.assertEqual(ctx3.exception.code, "DISALLOWED_HOST_HEADER")

        with self.assertRaises(SecurityViolationError) as ctx4:
            validate_inbound_localhost_request(
                host_header="rebind.attacker.invalid:8000",
                origin_header=None,
                referer_header=None,
                method="GET",
                allowed_frontend_origins=["http://127.0.0.1:8000"],
            )
        self.assertEqual(ctx4.exception.code, "DISALLOWED_HOST_HEADER")

        with self.assertRaises(SecurityViolationError) as ctx5:
            validate_inbound_localhost_request(
                host_header="127.0.0.1:8000",
                origin_header="http://evil.example.com:8000",
                referer_header=None,
                method="POST",
                allowed_frontend_origins=["http://127.0.0.1:8000"],
            )
        self.assertEqual(ctx5.exception.code, "DISALLOWED_ORIGIN")

        validate_inbound_localhost_request(
            host_header="127.0.0.1:8000",
            origin_header="http://127.0.0.1:8000",
            referer_header=None,
            method="POST",
            allowed_frontend_origins=["http://127.0.0.1:8000"],
        )

    def test_csv_formula_neutralization_and_hostile_html_export_escaping(self) -> None:
        self.assertEqual(neutralize_csv_cell("=CMD|' /C calc'!A0"), "'=CMD|' /C calc'!A0")
        self.assertEqual(neutralize_csv_cell("+SUM(A1:A10)"), "'+SUM(A1:A10)")
        self.assertEqual(neutralize_csv_cell("-2+3"), "'-2+3")
        self.assertEqual(neutralize_csv_cell("@HYPERLINK(\"http://evil\")"), "'@HYPERLINK(\"http://evil\")")
        self.assertEqual(neutralize_csv_cell(" \t=1+1"), "' \t=1+1")
        self.assertEqual(neutralize_csv_cell(-1.25), -1.25)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "sec_export.db")
            repo = Repository(db_path=db_path)
            worker = AuditWorkerService(repo=repo, start_background_loop=False)
            exporter = ExportService(repo=repo)

            hostile_name = "=CMD|' /C calc'!A0 <script>alert('xss_title')</script>"
            audit = worker.create_audit(
                AuditCreateRequest(
                    name=hostile_name,
                    reference_profile_id="prof_demo_reference",
                    target_profile_id="prof_demo_target_substituted",
                    score_mode=ScoreMode.LOCAL_LOGIT_RANK,
                    n_prompts=4,
                    m_reference=2,
                    start_immediately=False,
                )
            )
            aid = audit["id"]
            worker.start_audit(aid, run_synchronously=True)

            prompt_scores = repo.list_prompt_scores(aid)
            first_prompt_id = prompt_scores[0]["prompt_id"]
            conn = repo._conn()
            with conn:
                conn.execute(
                    "UPDATE audits SET name = ? WHERE id = ?;",
                    (hostile_name, aid),
                )
                conn.execute(
                    "UPDATE sample_responses SET response_text = ? WHERE audit_id = ? AND prompt_id = ? AND role = 'target';",
                    ("=IMPORTXML() <img src=x onerror=alert('pwned')>", aid, first_prompt_id),
                )
            conn.close()

            csv_text, _, _ = exporter.export_audit(aid, fmt="csv", output_path=str(Path(tmp) / "r.csv"))
            html_text, _, _ = exporter.export_audit(aid, fmt="html", output_path=str(Path(tmp) / "r.html"))
            json_text, _, _ = exporter.export_audit(aid, fmt="json", output_path=str(Path(tmp) / "r.json"))

            rows = list(csv.DictReader(io.StringIO(csv_text)))
            self.assertEqual(len(rows), 4)
            self.assertTrue(rows[0]["audit_name"].startswith("'=CMD|"))
            for idx, r in enumerate(rows):
                db_ps = prompt_scores[idx]
                self.assertEqual(r["prompt_id"], db_ps["prompt_id"])
                self.assertAlmostEqual(float(r["z_uniform"]), db_ps["z_uniform"], places=6)

            self.assertNotIn("<script>alert('xss_title')</script>", html_text)
            self.assertIn("&lt;script&gt;alert(&#x27;xss_title&#x27;)&lt;/script&gt;", html_text)
            self.assertNotIn("<img src=x onerror=alert('pwned')>", html_text)
            self.assertIn("&lt;img src=x onerror=alert(&#x27;pwned&#x27;)&gt;", html_text)
            self.assertIn("FULL_RESPONSES_PRESENT", html_text)

            json_before = json.loads(json_text)
            self.assertEqual(json_before["evidence_retention"]["status"], "FULL_RESPONSES_PRESENT")
            self.assertGreater(json_before["evidence_retention"]["stored_response_count"], 0)

            purge_res = repo.purge_audit_evidence(aid)
            self.assertGreater(purge_res["purged_response_count"], 0)
            self.assertTrue(purge_res["manifest_preserved"])

            html_after, _, _ = exporter.export_audit(aid, fmt="html", output_path=str(Path(tmp) / "r2.html"))
            json_after_text, _, _ = exporter.export_audit(aid, fmt="json", output_path=str(Path(tmp) / "r2.json"))
            json_after = json.loads(json_after_text)

            self.assertEqual(json_after["evidence_retention"]["status"], "PURGED_OR_EMPTY_MANIFEST_PRESERVED")
            self.assertEqual(json_after["evidence_retention"]["stored_response_count"], 0)
            self.assertEqual(len(json_after["prompt_scores"]), 4)
            self.assertEqual(json_after["audit"]["verdict"], json_before["audit"]["verdict"])
            self.assertIn("PURGED_OR_EMPTY_MANIFEST_PRESERVED", html_after)
