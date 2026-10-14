"""Deterministic SQLite schema migrations and baseline seed runner for ModelTrace."""

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modeltrace.config import canonical_json, get_env_settings, sha256_hex, utc_now_iso
from modeltrace.services.probes import BUILTIN_PROBES

MIGRATIONS: List[Tuple[str, str, str]] = [
    (
        "001_initial_schema",
        "Create core tables for profiles, immutable probe revisions, calibrations, audits, attempts, scores, events, exports, and settings",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            allowed_outbound_origins_json TEXT NOT NULL,
            allowed_local_inference_origins_json TEXT NOT NULL,
            retention_days INTEGER NOT NULL DEFAULT 90,
            default_max_duration_seconds INTEGER NOT NULL DEFAULT 600,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            adapter_type TEXT NOT NULL,
            base_url TEXT,
            api_key_env_var TEXT,
            model_identifier TEXT NOT NULL,
            checkpoint_path TEXT,
            checkpoint_revision TEXT DEFAULT 'main',
            tokenizer_path TEXT,
            chat_template TEXT,
            precision TEXT NOT NULL DEFAULT 'float32',
            device TEXT NOT NULL DEFAULT 'cpu',
            synthetic_scenario TEXT DEFAULT 'genuine',
            mixed_routing_bad_fraction REAL NOT NULL DEFAULT 0.5,
            timeout_seconds REAL NOT NULL DEFAULT 15.0,
            max_retries INTEGER NOT NULL DEFAULT 2,
            extra_config_json TEXT NOT NULL DEFAULT '{}',
            is_demo INTEGER NOT NULL DEFAULT 0,
            last_capability_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS probe_revisions (
            id TEXT PRIMARY KEY,
            suite_id TEXT NOT NULL,
            revision_number INTEGER NOT NULL,
            parent_revision_id TEXT,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            prompts_json TEXT NOT NULL,
            prompt_count INTEGER NOT NULL,
            families_json TEXT NOT NULL,
            validation_label TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(suite_id, revision_number)
        );

        CREATE TABLE IF NOT EXISTS calibrations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            score_mode TEXT NOT NULL,
            probe_revision_id TEXT NOT NULL,
            probe_revision_hash TEXT NOT NULL,
            n_prompts INTEGER NOT NULL,
            m_reference INTEGER NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            alpha REAL NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            reference_model_descriptor TEXT NOT NULL,
            is_demo INTEGER NOT NULL DEFAULT 0,
            provenance_notes TEXT NOT NULL DEFAULT '',
            alternatives_json TEXT NOT NULL,
            min_empirical_power REAL NOT NULL,
            min_trials INTEGER NOT NULL,
            calibration_seed INTEGER NOT NULL,
            artifact_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (probe_revision_id) REFERENCES probe_revisions(id)
        );

        CREATE TABLE IF NOT EXISTS audits (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            reference_profile_id TEXT NOT NULL,
            target_profile_id TEXT NOT NULL,
            scoring_reference_profile_id TEXT,
            probe_revision_id TEXT NOT NULL,
            probe_revision_hash TEXT NOT NULL,
            calibration_id TEXT,
            score_mode TEXT NOT NULL,
            is_demo INTEGER NOT NULL DEFAULT 0,
            n_prompts INTEGER NOT NULL,
            m_reference INTEGER NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            alpha REAL NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            mc_simulations INTEGER NOT NULL,
            include_secondary_tests INTEGER NOT NULL DEFAULT 0,
            max_target_attempts INTEGER NOT NULL,
            max_reference_attempts INTEGER NOT NULL,
            max_total_attempts INTEGER NOT NULL,
            max_duration_seconds INTEGER NOT NULL,
            master_seed INTEGER NOT NULL,
            seeds_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            execution_state TEXT NOT NULL,
            verdict TEXT NOT NULL,
            verdict_reason TEXT NOT NULL DEFAULT '',
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            resume_count INTEGER NOT NULL DEFAULT 0,
            uncertain_prior_attempts INTEGER NOT NULL DEFAULT 0,
            local_scoring_compute_ms REAL NOT NULL DEFAULT 0.0,
            summary_json TEXT,
            integrity_hash TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (reference_profile_id) REFERENCES profiles(id),
            FOREIGN KEY (target_profile_id) REFERENCES profiles(id),
            FOREIGN KEY (probe_revision_id) REFERENCES probe_revisions(id)
        );

        CREATE TABLE IF NOT EXISTS sample_attempts (
            id TEXT PRIMARY KEY,
            audit_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            attempt_number INTEGER NOT NULL,
            profile_id TEXT NOT NULL,
            status TEXT NOT NULL,
            http_status INTEGER,
            error_code TEXT,
            error_detail TEXT,
            latency_ms REAL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (audit_id) REFERENCES audits(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_attempts_audit_prompt_role
            ON sample_attempts(audit_id, prompt_id, role, sample_index);

        CREATE TABLE IF NOT EXISTS sample_responses (
            id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE,
            audit_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            role TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            response_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            unicode_length INTEGER NOT NULL,
            finish_reason TEXT,
            declared_model TEXT,
            observable_meta_json TEXT NOT NULL DEFAULT '{}',
            raw_score REAL,
            normalized_log_likelihood REAL,
            scoring_compute_ms REAL NOT NULL DEFAULT 0.0,
            response_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (attempt_id) REFERENCES sample_attempts(id) ON DELETE CASCADE,
            FOREIGN KEY (audit_id) REFERENCES audits(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_responses_audit_prompt
            ON sample_responses(audit_id, prompt_id, role);

        CREATE TABLE IF NOT EXISTS prompt_scores (
            id TEXT PRIMARY KEY,
            audit_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_index INTEGER NOT NULL,
            family TEXT NOT NULL,
            score_mode TEXT NOT NULL,
            m_reference INTEGER NOT NULL,
            reference_scores_json TEXT NOT NULL,
            target_score REAL NOT NULL,
            a_below INTEGER NOT NULL,
            b_ties INTEGER NOT NULL,
            j_draw INTEGER NOT NULL,
            u_draw REAL NOT NULL,
            z_uniform REAL NOT NULL,
            score_shift REAL NOT NULL,
            target_norm_ll REAL,
            ref_mean_norm_ll REAL,
            local_scoring_compute_ms REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL,
            UNIQUE(audit_id, prompt_id),
            FOREIGN KEY (audit_id) REFERENCES audits(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            audit_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'INFO',
            message TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (audit_id) REFERENCES audits(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_audit_created
            ON audit_events(audit_id, created_at);

        CREATE TABLE IF NOT EXISTS exports (
            id TEXT PRIMARY KEY,
            audit_id TEXT NOT NULL,
            export_format TEXT NOT NULL,
            file_path TEXT,
            content_hash TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (audit_id) REFERENCES audits(id) ON DELETE CASCADE
        );
        """,
    ),
    (
        "002_stage2_settings_and_pricing",
        "Add display_timezone, default screening parameters, and pricing_assumptions_json to app_settings",
        """
        ALTER TABLE app_settings ADD COLUMN display_timezone TEXT NOT NULL DEFAULT 'UTC';
        ALTER TABLE app_settings ADD COLUMN default_n_prompts INTEGER NOT NULL DEFAULT 20;
        ALTER TABLE app_settings ADD COLUMN default_m_reference INTEGER NOT NULL DEFAULT 8;
        ALTER TABLE app_settings ADD COLUMN default_alpha REAL NOT NULL DEFAULT 0.01;
        ALTER TABLE app_settings ADD COLUMN pricing_assumptions_json TEXT NOT NULL DEFAULT '{}';
        """,
    ),
    (
        "003_stage3_calibration_identity_binding",
        "Add full reference/scorer identity and statistical engine version binding to calibrations",
        """
        ALTER TABLE calibrations ADD COLUMN reference_identity_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE calibrations ADD COLUMN stat_engine_method TEXT NOT NULL DEFAULT 'cramer_von_mises_randomized_rank_mc';
        ALTER TABLE calibrations ADD COLUMN stat_engine_version TEXT NOT NULL DEFAULT '1.0.0';
        """,
    ),
]


class _SyncedSqliteConnection(sqlite3.Connection):
    """SQLite connection that transparently mirrors writes to a FUSE/9p target file on commit/close when needed."""

    _sync_target_path: Optional[str] = None

    def _sync_to_target(self) -> None:
        if self._sync_target_path:
            try:
                raw = self.serialize()
                target = Path(self._sync_target_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            except Exception:
                pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        res = super().__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None:
            self._sync_to_target()
        return res

    def close(self) -> None:
        self._sync_to_target()
        super().close()


_FS_WAL_SUPPORT_CACHE: Dict[str, bool] = {}


def _supports_native_wal(dir_path: str) -> bool:
    """Detect whether the target directory's filesystem supports native SQLite WAL mmap/locking."""
    if dir_path in _FS_WAL_SUPPORT_CACHE:
        return _FS_WAL_SUPPORT_CACHE[dir_path]
    probe_file = os.path.join(dir_path, f".wal_probe_{os.getpid()}.db")
    supported = True
    try:
        c = sqlite3.connect(probe_file, timeout=2.0)
        try:
            c.execute("PRAGMA journal_mode = WAL;")
            c.execute("CREATE TABLE _t (x INTEGER);")
            c.execute("INSERT INTO _t VALUES (1);")
            c.commit()
        finally:
            c.close()
    except sqlite3.OperationalError:
        supported = False
    finally:
        for ext in ("", "-wal", "-shm"):
            try:
                if os.path.exists(probe_file + ext):
                    os.remove(probe_file + ext)
            except OSError:
                pass
    _FS_WAL_SUPPORT_CACHE[dir_path] = supported
    return supported


def _resolve_sqlite_runtime_path(db_path: str) -> Tuple[str, Optional[str]]:
    """Use db_path directly when native WAL is supported; fall back to /tmp sync on 9p/FUSE mounts."""
    abs_target = os.path.abspath(db_path)
    parent_dir = str(Path(abs_target).parent)
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    if not _supports_native_wal(parent_dir):
        digest = hashlib.sha256(abs_target.encode("utf-8")).hexdigest()[:12]
        runtime_dir = Path("/tmp/modeltrace_dbs")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = str(runtime_dir / f"{digest}_{os.path.basename(abs_target)}")
        if (
            not os.path.exists(runtime_path)
            and os.path.isfile(abs_target)
            and os.path.getsize(abs_target) > 0
        ):
            shutil.copyfile(abs_target, runtime_path)
        return runtime_path, abs_target
    return abs_target, None


def get_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL mode, foreign keys, and Row factory."""
    runtime_path, sync_target = _resolve_sqlite_runtime_path(db_path)
    conn: _SyncedSqliteConnection = sqlite3.connect(
        runtime_path,
        timeout=30.0,
        check_same_thread=False,
        factory=_SyncedSqliteConnection,
    )
    conn._sync_target_path = sync_target
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def run_migrations(db_path: str) -> List[str]:
    """Apply all pending SQLite schema migrations and seed built-in baseline records idempotently."""
    conn = get_sqlite_connection(db_path)
    applied_now: List[str] = []
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
            cur = conn.execute("SELECT version FROM schema_migrations ORDER BY version;")
            existing = {row["version"] for row in cur.fetchall()}

            for version, description, sql_script in MIGRATIONS:
                if version not in existing:
                    conn.executescript(sql_script)
                    conn.execute("PRAGMA foreign_keys = ON;")
                    conn.execute(
                        "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?);",
                        (version, description, utc_now_iso()),
                    )
                    applied_now.append(version)

            _seed_defaults_if_needed(conn)
    finally:
        conn.close()
    return applied_now


def _seed_defaults_if_needed(conn: sqlite3.Connection) -> None:
    """Ensure singleton app_settings, built-in probe revision, and DEMO profiles exist."""
    env = get_env_settings()
    now = utc_now_iso()

    row = conn.execute("SELECT id FROM app_settings WHERE id = 1;").fetchone()
    if not row:
        conn.execute(
            """
            INSERT INTO app_settings (
                id, allowed_outbound_origins_json, allowed_local_inference_origins_json,
                retention_days, default_max_duration_seconds, display_timezone,
                default_n_prompts, default_m_reference, default_alpha,
                pricing_assumptions_json, updated_at
            ) VALUES (1, ?, ?, 90, 600, 'UTC', 20, 8, 0.01, '{}', ?);
            """,
            (
                json.dumps(env.default_allowed_outbound_origins),
                json.dumps(env.default_allowed_local_origins),
                now,
            ),
        )

    builtin_rev_id = "rev_builtin_screening_v1"
    rev_row = conn.execute(
        "SELECT id FROM probe_revisions WHERE id = ?;", (builtin_rev_id,)
    ).fetchone()
    if not rev_row:
        families = sorted({p["family"] for p in BUILTIN_PROBES})
        prompts_canonical = canonical_json(BUILTIN_PROBES)
        content_hash = sha256_hex(prompts_canonical)
        conn.execute(
            """
            INSERT INTO probe_revisions (
                id, suite_id, revision_number, parent_revision_id, name, description,
                prompts_json, prompt_count, families_json, validation_label, content_hash, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                builtin_rev_id,
                "suite_builtin_screening",
                1,
                "ModelTrace Core Screening Suite (v1)",
                "Original 24-prompt multi-family screening suite across extraction, reasoning, code, multilingual, and formatting tasks. Labeled unvalidated for universal identity discrimination.",
                prompts_canonical,
                len(BUILTIN_PROBES),
                json.dumps(families),
                "UNVALIDATED_FOR_IDENTITY_DISCRIMINATION",
                content_hash,
                now,
            ),
        )

    demo_profiles: List[Dict[str, object]] = [
        {
            "id": "prof_demo_reference",
            "name": "[DEMO] Trusted Reference Baseline (Llama-3-8B-Sim)",
            "description": "DEMO synthetic reference generator & local logit-rank scorer.",
            "adapter_type": "synthetic_demo",
            "model_identifier": "demo-llama-3-8b-instruct-fp16",
            "synthetic_scenario": "genuine",
            "mixed_routing_bad_fraction": 0.0,
            "is_demo": 1,
        },
        {
            "id": "prof_demo_target_genuine",
            "name": "[DEMO] Target Endpoint — Genuine Match",
            "description": "DEMO synthetic target matching the reference distribution.",
            "adapter_type": "synthetic_demo",
            "model_identifier": "demo-llama-3-8b-instruct-fp16",
            "synthetic_scenario": "genuine",
            "mixed_routing_bad_fraction": 0.0,
            "is_demo": 1,
        },
        {
            "id": "prof_demo_target_substituted",
            "name": "[DEMO] Target Endpoint — Substituted Distilled/Quantized Model",
            "description": "DEMO synthetic target serving a shifted substitute distribution.",
            "adapter_type": "synthetic_demo",
            "model_identifier": "demo-llama-3-8b-instruct-int4-sub",
            "synthetic_scenario": "substituted",
            "mixed_routing_bad_fraction": 1.0,
            "is_demo": 1,
        },
        {
            "id": "prof_demo_target_mixed",
            "name": "[DEMO] Target Endpoint — Mixed Routing (45% Substitute Pool)",
            "description": "DEMO synthetic target routing 45% of requests to a cheaper substitute.",
            "adapter_type": "synthetic_demo",
            "model_identifier": "demo-llama-3-8b-instruct-mixed-pool",
            "synthetic_scenario": "mixed_routing",
            "mixed_routing_bad_fraction": 0.45,
            "is_demo": 1,
        },
    ]

    for dp in demo_profiles:
        exists = conn.execute("SELECT id FROM profiles WHERE id = ?;", (dp["id"],)).fetchone()
        if not exists:
            conn.execute(
                """
                INSERT INTO profiles (
                    id, name, description, adapter_type, base_url, api_key_env_var,
                    model_identifier, checkpoint_path, checkpoint_revision, tokenizer_path,
                    chat_template, precision, device, synthetic_scenario,
                    mixed_routing_bad_fraction, timeout_seconds, max_retries,
                    extra_config_json, is_demo, last_capability_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, 'main', NULL, NULL, 'float32', 'cpu', ?, ?, 15.0, 2, '{}', ?, NULL, ?, ?);
                """,
                (
                    dp["id"],
                    dp["name"],
                    dp["description"],
                    dp["adapter_type"],
                    dp["model_identifier"],
                    dp["synthetic_scenario"],
                    dp["mixed_routing_bad_fraction"],
                    dp["is_demo"],
                    now,
                    now,
                ),
            )
