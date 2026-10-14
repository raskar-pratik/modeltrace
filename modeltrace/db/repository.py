"""Transactional SQLite repository for ModelTrace entities."""

import datetime
import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from modeltrace.config import canonical_json, sha256_hex, utc_now_iso
from modeltrace.db.migrations import get_sqlite_connection, run_migrations
from modeltrace.schemas import (
    AdapterType,
    AttemptStatus,
    AuditVerdict,
    CalibrationImportRequest,
    ExecutionState,
    ProbePromptItem,
    ProbeRevisionCreate,
    ProfileCreate,
    ProfileUpdate,
    SettingsUpdateRequest,
)
from modeltrace.services.stats import (
    STAT_ENGINE_METHOD,
    STAT_ENGINE_VERSION,
    build_reference_scorer_identity,
    validate_and_recompute_calibration_alternatives,
)
from modeltrace.utils.security import canonicalize_origin, redact_secrets, redact_text


class Repository:
    """Data access and persistence coordinator for ModelTrace."""

    def __init__(self, db_path: str, auto_migrate: bool = True) -> None:
        self.db_path = db_path
        if auto_migrate:
            run_migrations(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_sqlite_connection(self.db_path)

    # ------------------------------------------------------------------
    # Settings & Retention
    # ------------------------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM app_settings WHERE id = 1;").fetchone()
            if not row:
                run_migrations(self.db_path)
                row = conn.execute("SELECT * FROM app_settings WHERE id = 1;").fetchone()
            keys = row.keys()
            pricing_raw = (
                row["pricing_assumptions_json"]
                if "pricing_assumptions_json" in keys and row["pricing_assumptions_json"]
                else "{}"
            )
            pricing = json.loads(pricing_raw)
            if not isinstance(pricing, dict):
                pricing = {}
            return {
                "allowed_outbound_origins": json.loads(row["allowed_outbound_origins_json"]),
                "allowed_local_inference_origins": json.loads(
                    row["allowed_local_inference_origins_json"]
                ),
                "retention_days": int(row["retention_days"]),
                "default_max_duration_seconds": int(row["default_max_duration_seconds"]),
                "display_timezone": (
                    row["display_timezone"] if "display_timezone" in keys else "UTC"
                ),
                "default_n_prompts": (
                    int(row["default_n_prompts"]) if "default_n_prompts" in keys else 20
                ),
                "default_m_reference": (
                    int(row["default_m_reference"]) if "default_m_reference" in keys else 8
                ),
                "default_alpha": (
                    float(row["default_alpha"]) if "default_alpha" in keys else 0.01
                ),
                "pricing_assumptions": {
                    "reference_cost_per_1k_tokens_usd": pricing.get(
                        "reference_cost_per_1k_tokens_usd"
                    ),
                    "target_cost_per_1k_tokens_usd": pricing.get(
                        "target_cost_per_1k_tokens_usd"
                    ),
                    "estimated_prompt_tokens_per_call": int(
                        pricing.get("estimated_prompt_tokens_per_call", 64)
                    ),
                },
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def update_settings(self, req: SettingsUpdateRequest) -> Dict[str, Any]:
        current = self.get_settings()
        outbound = (
            [canonicalize_origin(o) for o in req.allowed_outbound_origins if o.strip()]
            if req.allowed_outbound_origins is not None
            else current["allowed_outbound_origins"]
        )
        local_origins = (
            [canonicalize_origin(o) for o in req.allowed_local_inference_origins if o.strip()]
            if req.allowed_local_inference_origins is not None
            else current["allowed_local_inference_origins"]
        )
        retention = (
            req.retention_days if req.retention_days is not None else current["retention_days"]
        )
        max_dur = (
            req.default_max_duration_seconds
            if req.default_max_duration_seconds is not None
            else current["default_max_duration_seconds"]
        )
        tz = (
            req.display_timezone.strip()
            if req.display_timezone is not None
            else current["display_timezone"]
        )
        def_n = (
            req.default_n_prompts
            if req.default_n_prompts is not None
            else current["default_n_prompts"]
        )
        def_m = (
            req.default_m_reference
            if req.default_m_reference is not None
            else current["default_m_reference"]
        )
        def_alpha = (
            req.default_alpha if req.default_alpha is not None else current["default_alpha"]
        )
        pricing = (
            req.pricing_assumptions.dict()
            if req.pricing_assumptions is not None
            else current["pricing_assumptions"]
        )

        now = utc_now_iso()
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE app_settings
                    SET allowed_outbound_origins_json = ?,
                        allowed_local_inference_origins_json = ?,
                        retention_days = ?,
                        default_max_duration_seconds = ?,
                        display_timezone = ?,
                        default_n_prompts = ?,
                        default_m_reference = ?,
                        default_alpha = ?,
                        pricing_assumptions_json = ?,
                        updated_at = ?
                    WHERE id = 1;
                    """,
                    (
                        json.dumps(outbound),
                        json.dumps(local_origins),
                        retention,
                        max_dur,
                        tz,
                        def_n,
                        def_m,
                        def_alpha,
                        json.dumps(pricing),
                        now,
                    ),
                )
        finally:
            conn.close()
        return self.get_settings()

    def purge_audit_evidence(self, audit_id: str) -> Dict[str, Any]:
        """Purge stored raw sample_responses for a non-running audit while preserving its manifest and summary."""
        conn = self._conn()
        try:
            with conn:
                row = conn.execute(
                    "SELECT execution_state FROM audits WHERE id = ?;", (audit_id,)
                ).fetchone()
                if not row:
                    raise KeyError(f"Audit '{audit_id}' not found.")
                if row["execution_state"] == ExecutionState.RUNNING.value:
                    raise ValueError(
                        "Cannot purge evidence for an actively RUNNING audit. Cancel it first."
                    )
                cur = conn.execute(
                    "DELETE FROM sample_responses WHERE audit_id = ?;", (audit_id,)
                )
                purged_count = int(cur.rowcount or 0)
        finally:
            conn.close()

        self.append_audit_event(
            audit_id,
            "EVIDENCE_PURGED",
            f"Operator confirmed evidence purge: deleted {purged_count} stored raw response record(s) while preserving immutable audit manifest and statistical summary.",
            level="WARNING",
            details={"purged_response_count": purged_count},
        )
        return {
            "audit_id": audit_id,
            "purged_response_count": purged_count,
            "manifest_preserved": True,
        }

    def run_retention_cleanup(
        self,
        older_than_days: Optional[int] = None,
        preserve_manifests: bool = True,
    ) -> Dict[str, Any]:
        """Purge expired evidence or finished audits older than retention_days (never touches RUNNING jobs)."""
        settings = self.get_settings()
        days = older_than_days if older_than_days is not None else settings["retention_days"]
        cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
        cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        conn = self._conn()
        try:
            with conn:
                cur = conn.execute(
                    """
                    SELECT id FROM audits
                    WHERE execution_state != 'RUNNING' AND created_at < ?;
                    """,
                    (cutoff_iso,),
                )
                target_ids = [r["id"] for r in cur.fetchall()]
                purged_responses = 0
                deleted_audits = 0
                for aid in target_ids:
                    if preserve_manifests:
                        c_resp = conn.execute(
                            "DELETE FROM sample_responses WHERE audit_id = ?;", (aid,)
                        )
                        purged_responses += int(c_resp.rowcount or 0)
                    else:
                        conn.execute("DELETE FROM audits WHERE id = ?;", (aid,))
                        deleted_audits += 1
                return {
                    "retention_days": days,
                    "cutoff_utc": cutoff_iso,
                    "preserve_manifests": preserve_manifests,
                    "affected_audit_count": len(target_ids),
                    "affected_audit_ids": target_ids,
                    "purged_response_count": purged_responses,
                    "deleted_audit_count": deleted_audits,
                }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------
    def _row_to_profile(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "adapter_type": row["adapter_type"],
            "base_url": row["base_url"],
            "api_key_env_var": row["api_key_env_var"],
            "model_identifier": row["model_identifier"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_revision": row["checkpoint_revision"],
            "tokenizer_path": row["tokenizer_path"],
            "chat_template": row["chat_template"],
            "precision": row["precision"],
            "device": row["device"],
            "synthetic_scenario": row["synthetic_scenario"],
            "mixed_routing_bad_fraction": float(row["mixed_routing_bad_fraction"]),
            "timeout_seconds": float(row["timeout_seconds"]),
            "max_retries": int(row["max_retries"]),
            "extra_config": json.loads(row["extra_config_json"] or "{}"),
            "is_demo": bool(row["is_demo"]),
            "last_capability": (
                json.loads(row["last_capability_json"]) if row["last_capability_json"] else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_profiles(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM profiles ORDER BY created_at ASC;").fetchall()
            return [self._row_to_profile(r) for r in rows]
        finally:
            conn.close()

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM profiles WHERE id = ?;", (profile_id,)).fetchone()
            return self._row_to_profile(row) if row else None
        finally:
            conn.close()

    def create_profile(self, req: ProfileCreate) -> Dict[str, Any]:
        profile_id = f"prof_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        is_demo = 1 if req.adapter_type == AdapterType.SYNTHETIC_DEMO else 0
        name = req.name
        if is_demo and not name.startswith("[DEMO]"):
            name = f"[DEMO] {name}"
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO profiles (
                        id, name, description, adapter_type, base_url, api_key_env_var,
                        model_identifier, checkpoint_path, checkpoint_revision, tokenizer_path,
                        chat_template, precision, device, synthetic_scenario,
                        mixed_routing_bad_fraction, timeout_seconds, max_retries,
                        extra_config_json, is_demo, last_capability_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?);
                    """,
                    (
                        profile_id,
                        name,
                        req.description,
                        req.adapter_type.value,
                        req.base_url,
                        req.api_key_env_var,
                        req.model_identifier,
                        req.checkpoint_path,
                        req.checkpoint_revision,
                        req.tokenizer_path,
                        req.chat_template,
                        req.precision,
                        req.device,
                        req.synthetic_scenario.value if req.synthetic_scenario else "genuine",
                        req.mixed_routing_bad_fraction,
                        req.timeout_seconds,
                        req.max_retries,
                        json.dumps(redact_secrets(req.extra_config)),
                        is_demo,
                        now,
                        now,
                    ),
                )
        finally:
            conn.close()
        return self.get_profile(profile_id)  # type: ignore[return-value]

    def update_profile(self, profile_id: str, req: ProfileUpdate) -> Optional[Dict[str, Any]]:
        existing = self.get_profile(profile_id)
        if not existing:
            return None
        updates = req.dict(exclude_unset=True)
        merged = dict(existing)
        for k, v in updates.items():
            if k == "synthetic_scenario" and v is not None:
                merged[k] = v.value if hasattr(v, "value") else v
            elif k == "extra_config" and v is not None:
                merged[k] = redact_secrets(v)
            else:
                merged[k] = v
        if merged["is_demo"] and not str(merged["name"]).startswith("[DEMO]"):
            merged["name"] = f"[DEMO] {merged['name']}"
        now = utc_now_iso()
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE profiles
                    SET name = ?, description = ?, base_url = ?, api_key_env_var = ?,
                        model_identifier = ?, checkpoint_path = ?, checkpoint_revision = ?,
                        tokenizer_path = ?, chat_template = ?, precision = ?, device = ?,
                        synthetic_scenario = ?, mixed_routing_bad_fraction = ?,
                        timeout_seconds = ?, max_retries = ?, extra_config_json = ?,
                        updated_at = ?
                    WHERE id = ?;
                    """,
                    (
                        merged["name"],
                        merged["description"],
                        merged["base_url"],
                        merged["api_key_env_var"],
                        merged["model_identifier"],
                        merged["checkpoint_path"],
                        merged["checkpoint_revision"],
                        merged["tokenizer_path"],
                        merged["chat_template"],
                        merged["precision"],
                        merged["device"],
                        merged["synthetic_scenario"],
                        merged["mixed_routing_bad_fraction"],
                        merged["timeout_seconds"],
                        merged["max_retries"],
                        json.dumps(merged["extra_config"]),
                        now,
                        profile_id,
                    ),
                )
        finally:
            conn.close()
        return self.get_profile(profile_id)

    def record_profile_capability(
        self, profile_id: str, capability_report: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        sanitized = redact_secrets(capability_report)
        now = utc_now_iso()
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    "UPDATE profiles SET last_capability_json = ?, updated_at = ? WHERE id = ?;",
                    (json.dumps(sanitized), now, profile_id),
                )
        finally:
            conn.close()
        return self.get_profile(profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        conn = self._conn()
        try:
            with conn:
                ref_count = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt FROM audits
                    WHERE reference_profile_id = ? OR target_profile_id = ?
                       OR scoring_reference_profile_id = ?;
                    """,
                    (profile_id, profile_id, profile_id),
                ).fetchone()["cnt"]
                if ref_count > 0:
                    raise ValueError(
                        f"Cannot delete profile '{profile_id}' because it is referenced by {ref_count} persisted audit(s) (history is immutable)."
                    )
                cur = conn.execute("DELETE FROM profiles WHERE id = ?;", (profile_id,))
                return cur.rowcount > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Immutable Probe Revisions
    # ------------------------------------------------------------------
    def _row_to_probe_revision(
        self, row: sqlite3.Row, include_prompts: bool = True
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": row["id"],
            "suite_id": row["suite_id"],
            "revision_number": int(row["revision_number"]),
            "parent_revision_id": row["parent_revision_id"],
            "name": row["name"],
            "description": row["description"],
            "prompt_count": int(row["prompt_count"]),
            "families": json.loads(row["families_json"]),
            "validation_label": row["validation_label"],
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
        }
        if include_prompts:
            data["prompts"] = json.loads(row["prompts_json"])
        return data

    def list_probe_revisions(self, include_prompts: bool = False) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM probe_revisions ORDER BY created_at DESC, revision_number DESC;"
            ).fetchall()
            return [self._row_to_probe_revision(r, include_prompts=include_prompts) for r in rows]
        finally:
            conn.close()

    def get_probe_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM probe_revisions WHERE id = ?;", (revision_id,)
            ).fetchone()
            return self._row_to_probe_revision(row, include_prompts=True) if row else None
        finally:
            conn.close()

    def get_default_probe_revision(self) -> Dict[str, Any]:
        rev = self.get_probe_revision("rev_builtin_screening_v1")
        if rev:
            return rev
        revs = self.list_probe_revisions(include_prompts=True)
        if not revs:
            raise RuntimeError("No probe revisions found in database.")
        return revs[0]

    def create_probe_revision(self, req: ProbeRevisionCreate) -> Dict[str, Any]:
        """Create a new immutable probe revision. Never modifies existing revisions."""
        conn = self._conn()
        try:
            with conn:
                suite_id = f"suite_{uuid.uuid4().hex[:10]}"
                next_rev_num = 1
                if req.parent_revision_id:
                    parent = conn.execute(
                        "SELECT suite_id FROM probe_revisions WHERE id = ?;",
                        (req.parent_revision_id,),
                    ).fetchone()
                    if not parent:
                        raise ValueError(
                            f"Parent probe revision '{req.parent_revision_id}' does not exist."
                        )
                    suite_id = parent["suite_id"]
                    max_rev = conn.execute(
                        "SELECT MAX(revision_number) AS max_r FROM probe_revisions WHERE suite_id = ?;",
                        (suite_id,),
                    ).fetchone()["max_r"]
                    next_rev_num = int(max_rev or 0) + 1

                prompts_dicts = [p.dict() for p in req.prompts]
                prompts_canonical = canonical_json(prompts_dicts)
                content_hash = sha256_hex(prompts_canonical)
                families = sorted({p.family for p in req.prompts})
                rev_id = f"rev_{uuid.uuid4().hex[:12]}"
                now = utc_now_iso()

                conn.execute(
                    """
                    INSERT INTO probe_revisions (
                        id, suite_id, revision_number, parent_revision_id, name, description,
                        prompts_json, prompt_count, families_json, validation_label, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        rev_id,
                        suite_id,
                        next_rev_num,
                        req.parent_revision_id,
                        req.name,
                        req.description,
                        prompts_canonical,
                        len(prompts_dicts),
                        json.dumps(families),
                        "UNVALIDATED_FOR_IDENTITY_DISCRIMINATION",
                        content_hash,
                        now,
                    ),
                )
        finally:
            conn.close()
        return self.get_probe_revision(rev_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Calibrations
    # ------------------------------------------------------------------
    def _row_to_calibration(self, row: sqlite3.Row) -> Dict[str, Any]:
        keys = row.keys()
        ref_id_raw = (
            row["reference_identity_json"]
            if "reference_identity_json" in keys and row["reference_identity_json"]
            else "{}"
        )
        try:
            ref_identity = json.loads(ref_id_raw)
        except Exception:
            ref_identity = {}
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "score_mode": row["score_mode"],
            "probe_revision_id": row["probe_revision_id"],
            "probe_revision_hash": row["probe_revision_hash"],
            "n_prompts": int(row["n_prompts"]),
            "m_reference": int(row["m_reference"]),
            "max_output_tokens": int(row["max_output_tokens"]),
            "alpha": float(row["alpha"]),
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "reference_model_descriptor": row["reference_model_descriptor"],
            "reference_identity": ref_identity if isinstance(ref_identity, dict) else {},
            "stat_engine_method": (
                row["stat_engine_method"]
                if "stat_engine_method" in keys and row["stat_engine_method"]
                else STAT_ENGINE_METHOD
            ),
            "stat_engine_version": (
                row["stat_engine_version"]
                if "stat_engine_version" in keys and row["stat_engine_version"]
                else STAT_ENGINE_VERSION
            ),
            "is_demo": bool(row["is_demo"]),
            "provenance_notes": row["provenance_notes"],
            "alternatives": json.loads(row["alternatives_json"]),
            "min_empirical_power": float(row["min_empirical_power"]),
            "min_trials": int(row["min_trials"]),
            "calibration_seed": int(row["calibration_seed"]),
            "artifact_hash": row["artifact_hash"],
            "created_at": row["created_at"],
        }

    def list_calibrations(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM calibrations ORDER BY created_at DESC;").fetchall()
            return [self._row_to_calibration(r) for r in rows]
        finally:
            conn.close()

    def get_calibration(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM calibrations WHERE id = ?;", (calibration_id,)
            ).fetchone()
            return self._row_to_calibration(row) if row else None
        finally:
            conn.close()

    def import_calibration(self, req: CalibrationImportRequest) -> Dict[str, Any]:
        """Validate probe revision hash, recompute power & Wilson/Clopper-Pearson bounds from counts, and bind reference identity."""
        probe_rev = self.get_probe_revision(req.probe_revision_id)
        if not probe_rev:
            raise ValueError(
                f"Calibration references unknown probe_revision_id '{req.probe_revision_id}'."
            )
        if probe_rev["content_hash"] != req.probe_revision_hash:
            raise ValueError(
                f"Calibration probe_revision_hash mismatch: expected '{probe_rev['content_hash']}', got '{req.probe_revision_hash}'."
            )
        if req.n_prompts > probe_rev["prompt_count"]:
            raise ValueError(
                f"Calibration n_prompts ({req.n_prompts}) exceeds probe revision prompt_count ({probe_rev['prompt_count']})."
            )

        raw_alts = [a.dict() for a in req.alternatives]
        recomputed = validate_and_recompute_calibration_alternatives(raw_alts)
        alts_dicts = recomputed["alternatives"]
        min_power = recomputed["min_empirical_power"]
        min_trials = recomputed["min_trials"]

        ref_identity: Dict[str, Any] = {}
        if req.reference_identity and isinstance(req.reference_identity, dict):
            ref_identity = dict(req.reference_identity)
        elif req.reference_profile_id:
            ref_prof = self.get_profile(req.reference_profile_id)
            if not ref_prof:
                raise ValueError(f"Calibration reference_profile_id '{req.reference_profile_id}' not found.")
            sc_prof = (
                self.get_profile(req.scoring_reference_profile_id)
                if req.scoring_reference_profile_id
                else ref_prof
            )
            ref_identity = build_reference_scorer_identity(ref_prof, sc_prof)

        cal_id = f"cal_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        name = req.name
        if req.is_demo and not name.startswith("[DEMO]"):
            name = f"[DEMO] {name}"

        artifact_payload = {
            "name": name,
            "score_mode": req.score_mode.value,
            "probe_revision_id": req.probe_revision_id,
            "probe_revision_hash": req.probe_revision_hash,
            "n_prompts": req.n_prompts,
            "m_reference": req.m_reference,
            "max_output_tokens": req.max_output_tokens,
            "alpha": req.alpha,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "reference_model_descriptor": req.reference_model_descriptor,
            "reference_identity": ref_identity,
            "stat_engine_method": req.stat_engine_method,
            "stat_engine_version": req.stat_engine_version,
            "is_demo": req.is_demo,
            "alternatives": alts_dicts,
            "calibration_seed": req.calibration_seed,
        }
        artifact_hash = sha256_hex(artifact_payload)

        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO calibrations (
                        id, name, description, score_mode, probe_revision_id, probe_revision_hash,
                        n_prompts, m_reference, max_output_tokens, alpha, temperature, top_p,
                        reference_model_descriptor, reference_identity_json, stat_engine_method,
                        stat_engine_version, is_demo, provenance_notes, alternatives_json,
                        min_empirical_power, min_trials, calibration_seed, artifact_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        cal_id,
                        name,
                        req.description,
                        req.score_mode.value,
                        req.probe_revision_id,
                        req.probe_revision_hash,
                        req.n_prompts,
                        req.m_reference,
                        req.max_output_tokens,
                        req.alpha,
                        req.temperature,
                        req.top_p,
                        req.reference_model_descriptor,
                        canonical_json(ref_identity),
                        req.stat_engine_method,
                        req.stat_engine_version,
                        1 if req.is_demo else 0,
                        req.provenance_notes,
                        canonical_json(alts_dicts),
                        min_power,
                        min_trials,
                        req.calibration_seed,
                        artifact_hash,
                        now,
                    ),
                )
        finally:
            conn.close()
        return self.get_calibration(cal_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Audits, Attempts, Responses, Scores, Events, Exports
    # ------------------------------------------------------------------
    def _row_to_audit(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "notes": row["notes"],
            "reference_profile_id": row["reference_profile_id"],
            "target_profile_id": row["target_profile_id"],
            "scoring_reference_profile_id": row["scoring_reference_profile_id"],
            "probe_revision_id": row["probe_revision_id"],
            "probe_revision_hash": row["probe_revision_hash"],
            "calibration_id": row["calibration_id"],
            "score_mode": row["score_mode"],
            "is_demo": bool(row["is_demo"]),
            "n_prompts": int(row["n_prompts"]),
            "m_reference": int(row["m_reference"]),
            "max_output_tokens": int(row["max_output_tokens"]),
            "alpha": float(row["alpha"]),
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "mc_simulations": int(row["mc_simulations"]),
            "include_secondary_tests": bool(row["include_secondary_tests"]),
            "max_target_attempts": int(row["max_target_attempts"]),
            "max_reference_attempts": int(row["max_reference_attempts"]),
            "max_total_attempts": int(row["max_total_attempts"]),
            "max_duration_seconds": int(row["max_duration_seconds"]),
            "master_seed": int(row["master_seed"]),
            "seeds": json.loads(row["seeds_json"]),
            "manifest": json.loads(row["manifest_json"]),
            "manifest_hash": row["manifest_hash"],
            "execution_state": row["execution_state"],
            "verdict": row["verdict"],
            "verdict_reason": row["verdict_reason"],
            "cancel_requested": bool(row["cancel_requested"]),
            "resume_count": int(row["resume_count"]),
            "uncertain_prior_attempts": int(row["uncertain_prior_attempts"]),
            "local_scoring_compute_ms": float(row["local_scoring_compute_ms"]),
            "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
            "integrity_hash": row["integrity_hash"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "updated_at": row["updated_at"],
        }

    def list_audits(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM audits ORDER BY created_at DESC;").fetchall()
            return [self._row_to_audit(r) for r in rows]
        finally:
            conn.close()

    def get_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM audits WHERE id = ?;", (audit_id,)).fetchone()
            return self._row_to_audit(row) if row else None
        finally:
            conn.close()

    def append_audit_event(
        self,
        audit_id: str,
        event_type: str,
        message: str,
        level: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        clean_msg = redact_text(message)
        clean_details = redact_secrets(details or {})
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO audit_events (id, audit_id, event_type, level, message, details_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        audit_id,
                        event_type,
                        level,
                        clean_msg,
                        json.dumps(clean_details),
                        now,
                    ),
                )
        finally:
            conn.close()
        return {
            "id": event_id,
            "audit_id": audit_id,
            "event_type": event_type,
            "level": level,
            "message": clean_msg,
            "details": clean_details,
            "created_at": now,
        }

    def list_audit_events(self, audit_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE audit_id = ? ORDER BY created_at ASC;",
                (audit_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "audit_id": r["audit_id"],
                    "event_type": r["event_type"],
                    "level": r["level"],
                    "message": r["message"],
                    "details": json.loads(r["details_json"] or "{}"),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def create_sample_attempt(
        self,
        audit_id: str,
        prompt_id: str,
        prompt_index: int,
        role: str,
        sample_index: int,
        attempt_number: int,
        profile_id: str,
    ) -> str:
        """Persist a sample attempt in IN_FLIGHT state BEFORE executing any network/model call."""
        attempt_id = f"att_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sample_attempts (
                        id, audit_id, prompt_id, prompt_index, role, sample_index,
                        attempt_number, profile_id, status, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        attempt_id,
                        audit_id,
                        prompt_id,
                        prompt_index,
                        role,
                        sample_index,
                        attempt_number,
                        profile_id,
                        AttemptStatus.IN_FLIGHT.value,
                        now,
                    ),
                )
        finally:
            conn.close()
        return attempt_id

    def complete_sample_attempt(
        self,
        attempt_id: str,
        audit_id: str,
        prompt_id: str,
        role: str,
        sample_index: int,
        response_text: str,
        token_count: int,
        unicode_length: int,
        finish_reason: Optional[str],
        declared_model: Optional[str],
        observable_meta: Dict[str, Any],
        raw_score: float,
        normalized_log_likelihood: Optional[float],
        scoring_compute_ms: float,
        latency_ms: float,
        http_status: int = 200,
        retry_count: int = 0,
    ) -> str:
        """Atomically mark attempt COMPLETED and persist the sample_responses record."""
        resp_id = f"rsp_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        clean_meta = redact_secrets(observable_meta)
        resp_hash = sha256_hex(
            {
                "audit_id": audit_id,
                "prompt_id": prompt_id,
                "role": role,
                "sample_index": sample_index,
                "response_text": response_text,
                "raw_score": round(raw_score, 9),
            }
        )
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE sample_attempts
                    SET status = ?, http_status = ?, latency_ms = ?, retry_count = ?, completed_at = ?
                    WHERE id = ?;
                    """,
                    (
                        AttemptStatus.COMPLETED.value,
                        http_status,
                        latency_ms,
                        retry_count,
                        now,
                        attempt_id,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO sample_responses (
                        id, attempt_id, audit_id, prompt_id, role, sample_index,
                        response_text, token_count, unicode_length, finish_reason,
                        declared_model, observable_meta_json, raw_score,
                        normalized_log_likelihood, scoring_compute_ms, response_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        resp_id,
                        attempt_id,
                        audit_id,
                        prompt_id,
                        role,
                        sample_index,
                        response_text,
                        token_count,
                        unicode_length,
                        finish_reason,
                        declared_model,
                        json.dumps(clean_meta),
                        raw_score,
                        normalized_log_likelihood,
                        scoring_compute_ms,
                        resp_hash,
                        now,
                    ),
                )
                conn.execute(
                    """
                    UPDATE audits
                    SET local_scoring_compute_ms = local_scoring_compute_ms + ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (scoring_compute_ms, now, audit_id),
                )
        finally:
            conn.close()
        return resp_id

    def fail_sample_attempt(
        self,
        attempt_id: str,
        error_code: str,
        error_detail: str,
        http_status: Optional[int] = None,
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
    ) -> None:
        now = utc_now_iso()
        clean_detail = redact_text(error_detail)
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE sample_attempts
                    SET status = ?, error_code = ?, error_detail = ?, http_status = ?,
                        latency_ms = ?, retry_count = ?, completed_at = ?
                    WHERE id = ?;
                    """,
                    (
                        AttemptStatus.FAILED.value,
                        error_code,
                        clean_detail,
                        http_status,
                        latency_ms,
                        retry_count,
                        now,
                        attempt_id,
                    ),
                )
        finally:
            conn.close()

    def get_attempt_counts(self, audit_id: str) -> Dict[str, int]:
        """Count persisted attempts (including retries and uncertain prior attempts) for budget enforcement."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT role, status, COUNT(*) AS cnt FROM sample_attempts WHERE audit_id = ? GROUP BY role, status;",
                (audit_id,),
            ).fetchall()
            ref_total = 0
            tgt_total = 0
            completed_ref = 0
            completed_tgt = 0
            failed_total = 0
            uncertain_total = 0
            for r in rows:
                cnt = int(r["cnt"])
                role = r["role"]
                st = r["status"]
                if role == "reference":
                    ref_total += cnt
                    if st == AttemptStatus.COMPLETED.value:
                        completed_ref += cnt
                elif role == "target":
                    tgt_total += cnt
                    if st == AttemptStatus.COMPLETED.value:
                        completed_tgt += cnt
                if st == AttemptStatus.FAILED.value:
                    failed_total += cnt
                elif st == AttemptStatus.UNCERTAIN_PRIOR_ATTEMPT.value:
                    uncertain_total += cnt

            return {
                "reference_attempts_used": ref_total,
                "target_attempts_used": tgt_total,
                "total_attempts_used": ref_total + tgt_total,
                "completed_reference_samples": completed_ref,
                "completed_target_samples": completed_tgt,
                "failed_attempts": failed_total,
                "uncertain_prior_attempts": uncertain_total,
            }
        finally:
            conn.close()

    def list_audit_responses(self, audit_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM sample_responses WHERE audit_id = ? ORDER BY prompt_id ASC, role ASC, sample_index ASC;",
                (audit_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "attempt_id": r["attempt_id"],
                    "audit_id": r["audit_id"],
                    "prompt_id": r["prompt_id"],
                    "role": r["role"],
                    "sample_index": int(r["sample_index"]),
                    "response_text": r["response_text"],
                    "token_count": int(r["token_count"]),
                    "unicode_length": int(r["unicode_length"]),
                    "finish_reason": r["finish_reason"],
                    "declared_model": r["declared_model"],
                    "observable_meta": json.loads(r["observable_meta_json"] or "{}"),
                    "raw_score": float(r["raw_score"]),
                    "normalized_log_likelihood": (
                        float(r["normalized_log_likelihood"])
                        if r["normalized_log_likelihood"] is not None
                        else None
                    ),
                    "scoring_compute_ms": float(r["scoring_compute_ms"]),
                    "response_hash": r["response_hash"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def list_audit_attempts(self, audit_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM sample_attempts WHERE audit_id = ? ORDER BY started_at ASC;",
                (audit_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "audit_id": r["audit_id"],
                    "prompt_id": r["prompt_id"],
                    "prompt_index": int(r["prompt_index"]),
                    "role": r["role"],
                    "sample_index": int(r["sample_index"]),
                    "attempt_number": int(r["attempt_number"]),
                    "profile_id": r["profile_id"],
                    "status": r["status"],
                    "http_status": r["http_status"],
                    "error_code": r["error_code"],
                    "error_detail": r["error_detail"],
                    "latency_ms": r["latency_ms"],
                    "retry_count": int(r["retry_count"]),
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def save_prompt_score(self, score_record: Dict[str, Any]) -> None:
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prompt_scores (
                        id, audit_id, prompt_id, prompt_index, family, score_mode,
                        m_reference, reference_scores_json, target_score, a_below, b_ties,
                        j_draw, u_draw, z_uniform, score_shift, target_norm_ll,
                        ref_mean_norm_ll, local_scoring_compute_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        score_record.get("id") or f"psc_{uuid.uuid4().hex[:12]}",
                        score_record["audit_id"],
                        score_record["prompt_id"],
                        score_record["prompt_index"],
                        score_record["family"],
                        score_record["score_mode"],
                        score_record["m_reference"],
                        canonical_json(score_record["reference_scores"]),
                        score_record["target_score"],
                        score_record["a_below"],
                        score_record["b_ties"],
                        score_record["j_draw"],
                        score_record["u_draw"],
                        score_record["z_uniform"],
                        score_record["score_shift"],
                        score_record.get("target_norm_ll"),
                        score_record.get("ref_mean_norm_ll"),
                        score_record.get("local_scoring_compute_ms", 0.0),
                        utc_now_iso(),
                    ),
                )
        finally:
            conn.close()

    def list_prompt_scores(self, audit_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM prompt_scores WHERE audit_id = ? ORDER BY prompt_index ASC;",
                (audit_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "audit_id": r["audit_id"],
                    "prompt_id": r["prompt_id"],
                    "prompt_index": int(r["prompt_index"]),
                    "family": r["family"],
                    "score_mode": r["score_mode"],
                    "m_reference": int(r["m_reference"]),
                    "reference_scores": json.loads(r["reference_scores_json"]),
                    "target_score": float(r["target_score"]),
                    "a_below": int(r["a_below"]),
                    "b_ties": int(r["b_ties"]),
                    "j_draw": int(r["j_draw"]),
                    "u_draw": float(r["u_draw"]),
                    "z_uniform": float(r["z_uniform"]),
                    "score_shift": float(r["score_shift"]),
                    "target_norm_ll": (
                        float(r["target_norm_ll"]) if r["target_norm_ll"] is not None else None
                    ),
                    "ref_mean_norm_ll": (
                        float(r["ref_mean_norm_ll"]) if r["ref_mean_norm_ll"] is not None else None
                    ),
                    "local_scoring_compute_ms": float(r["local_scoring_compute_ms"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def delete_audit(self, audit_id: str) -> bool:
        conn = self._conn()
        try:
            with conn:
                row = conn.execute(
                    "SELECT execution_state FROM audits WHERE id = ?;", (audit_id,)
                ).fetchone()
                if not row:
                    return False
                if row["execution_state"] == ExecutionState.RUNNING.value:
                    raise ValueError("Cannot delete an actively RUNNING audit. Cancel it first.")
                cur = conn.execute("DELETE FROM audits WHERE id = ?;", (audit_id,))
                return cur.rowcount > 0
        finally:
            conn.close()

    def record_export(
        self,
        audit_id: str,
        export_format: str,
        file_path: Optional[str],
        content_hash: str,
        byte_size: int,
    ) -> Dict[str, Any]:
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO exports (id, audit_id, export_format, file_path, content_hash, byte_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (exp_id, audit_id, export_format, file_path, content_hash, byte_size, now),
                )
        finally:
            conn.close()
        return {
            "id": exp_id,
            "audit_id": audit_id,
            "export_format": export_format,
            "file_path": file_path,
            "content_hash": content_hash,
            "byte_size": byte_size,
            "created_at": now,
        }

    def list_exports(self, audit_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM exports WHERE audit_id = ? ORDER BY created_at DESC;",
                (audit_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "audit_id": r["audit_id"],
                    "export_format": r["export_format"],
                    "file_path": r["file_path"],
                    "content_hash": r["content_hash"],
                    "byte_size": int(r["byte_size"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()
