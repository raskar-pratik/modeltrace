"""Tests for SQLite migrations, immutability of probe revisions and audit history, and retention."""

import tempfile
import unittest
from pathlib import Path

from modeltrace.db.migrations import run_migrations
from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AdapterType,
    AuditCreateRequest,
    ProbePromptItem,
    ProbeRevisionCreate,
    ProfileCreate,
    ScoreMode,
)
from modeltrace.services.probes import parse_probe_import_payload
from modeltrace.services.worker import AuditWorkerService


class TestMigrationsAndDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp_dir.name) / "test_modeltrace.db")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_clean_migrations_and_idempotency(self) -> None:
        first_applied = run_migrations(self.db_path)
        self.assertIn("001_initial_schema", first_applied)

        # Second invocation must be a no-op
        second_applied = run_migrations(self.db_path)
        self.assertEqual(second_applied, [])

        repo = Repository(self.db_path, auto_migrate=False)
        profiles = repo.list_profiles()
        self.assertGreaterEqual(len(profiles), 4)
        self.assertTrue(all(p["is_demo"] for p in profiles))

        default_rev = repo.get_default_probe_revision()
        self.assertEqual(default_rev["prompt_count"], 24)
        self.assertEqual(
            default_rev["validation_label"], "UNVALIDATED_FOR_IDENTITY_DISCRIMINATION"
        )
        self.assertEqual(
            sorted(default_rev["families"]),
            ["code", "extraction", "formatting", "multilingual", "reasoning"],
        )

    def test_immutable_probe_revisions_and_import_validation(self) -> None:
        repo = Repository(self.db_path)
        rev1 = repo.get_default_probe_revision()
        rev1_hash = rev1["content_hash"]

        # Create an edited revision referencing rev1 as parent
        edited_prompts = [
            ProbePromptItem(
                prompt_id="custom-01",
                family="reasoning",
                title="Custom Reasoning Prompt",
                prompt_text="Compute 7 * 13 and explain.",
            ),
            ProbePromptItem(
                prompt_id="custom-02",
                family="code",
                title="Custom Code Prompt",
                prompt_text="Write a function add(a, b).",
            ),
        ]
        rev2 = repo.create_probe_revision(
            ProbeRevisionCreate(
                name="Edited Screening Suite v2",
                description="Modified prompts without mutating v1",
                parent_revision_id=rev1["id"],
                prompts=edited_prompts,
            )
        )
        self.assertEqual(rev2["suite_id"], rev1["suite_id"])
        self.assertEqual(rev2["revision_number"], 2)
        self.assertNotEqual(rev2["content_hash"], rev1_hash)

        # Verify rev1 is completely untouched
        rev1_after = repo.get_probe_revision(rev1["id"])
        assert rev1_after is not None
        self.assertEqual(rev1_after["content_hash"], rev1_hash)
        self.assertEqual(rev1_after["prompt_count"], 24)

        # Verify duplicate prompt_id is rejected on JSONL import
        dup_jsonl = (
            '{"prompt_id": "dup-1", "family": "code", "title": "T1", "prompt_text": "P1"}\n'
            '{"prompt_id": "dup-1", "family": "code", "title": "T2", "prompt_text": "P2"}\n'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_probe_import_payload("jsonl", dup_jsonl)
        self.assertIn("Duplicate prompt_id", str(ctx.exception))

    def test_profile_deletion_blocked_when_referenced_by_audit(self) -> None:
        repo = Repository(self.db_path)
        worker = AuditWorkerService(repo=repo, start_background_loop=False)

        custom_prof = repo.create_profile(
            ProfileCreate(
                name="Temp Synthetic Target",
                adapter_type=AdapterType.SYNTHETIC_DEMO,
                model_identifier="temp-demo-model",
            )
        )

        audit = worker.create_audit(
            AuditCreateRequest(
                name="Audit Referencing Custom Profile",
                reference_profile_id="prof_demo_reference",
                target_profile_id=custom_prof["id"],
                score_mode=ScoreMode.LOCAL_LOGIT_RANK,
                n_prompts=2,
                m_reference=2,
            )
        )
        self.assertIsNotNone(audit["id"])

        with self.assertRaises(ValueError) as ctx:
            repo.delete_profile(custom_prof["id"])
        self.assertIn("history is immutable", str(ctx.exception))
