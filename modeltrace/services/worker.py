"""Database-backed single worker and audit lifecycle coordinator for ModelTrace.

Guarantees:
1. Heavy inference and statistical simulations run in a single background worker thread (never blocking HTTP handlers) or synchronously via CLI runner.
2. Persist sample_attempts in IN_FLIGHT status BEFORE making external calls.
3. Prevent double starts via atomic conditional state updates (`CREATED -> RUNNING`).
4. Startup recovery marks unfinished (`RUNNING`) jobs as `INTERRUPTED` and `IN_FLIGHT` attempts as `UNCERTAIN_PRIOR_ATTEMPT`—never silently repeats paid work.
5. Explicit resume reuses completed results, accounts conservatively for uncertain prior attempts against attempt caps, and records that exactly-once external calls cannot be promised.
6. Enforces hard caps on target attempts, reference attempts, total attempts (including retries), duration, and cooperative cancellation.
"""

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from modeltrace.adapters.base import AdapterError, GenerationResult, ScoringResult
from modeltrace.adapters.local_transformers import LocalTransformersAdapter
from modeltrace.adapters.openai_compatible import OpenAICompatibleAdapter
from modeltrace.adapters.synthetic_demo import SyntheticDemoAdapter
from modeltrace.config import canonical_json, sha256_hex, utc_now_iso
from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AdapterType,
    AttemptStatus,
    AuditCreateRequest,
    AuditPreviewRequest,
    AuditVerdict,
    CalibrationAlternativeResult,
    CalibrationImportRequest,
    DemoCalibrationCreateRequest,
    ExecutionState,
    ScoreMode,
    SyntheticScenario,
)
from modeltrace.services.stats import (
    STAT_ENGINE_METHOD,
    STAT_ENGINE_VERSION,
    build_reference_scorer_identity,
    check_calibration_compatibility,
    clopper_pearson_interval_95,
    compute_bootstrap_uncertainty,
    compute_prompt_rank_z,
    compute_unicode_length_score,
    derive_audit_seeds,
    evaluate_verdict,
    holm_step_down,
    run_cvm_monte_carlo_test,
    wilson_lower_bound_95,
)
from modeltrace.utils.security import redact_secrets, redact_text, validate_secret_env_var_name


def freeze_profile_snapshot(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Create an immutable, secret-free versioned snapshot of a profile for an audit manifest."""
    env_var = validate_secret_env_var_name(profile.get("api_key_env_var"))
    snap: Dict[str, Any] = {
        "snapshot_version": 1,
        "id": str(profile["id"]),
        "name": str(profile["name"]),
        "description": str(profile.get("description") or ""),
        "adapter_type": str(profile["adapter_type"]),
        "base_url": profile.get("base_url"),
        "api_key_env_var": env_var,
        "model_identifier": str(profile["model_identifier"]),
        "checkpoint_path": profile.get("checkpoint_path"),
        "checkpoint_revision": profile.get("checkpoint_revision") or "main",
        "tokenizer_path": profile.get("tokenizer_path"),
        "chat_template": profile.get("chat_template"),
        "precision": str(profile.get("precision") or "float32"),
        "device": str(profile.get("device") or "cpu"),
        "synthetic_scenario": profile.get("synthetic_scenario"),
        "mixed_routing_bad_fraction": float(profile.get("mixed_routing_bad_fraction", 0.5)),
        "timeout_seconds": float(profile.get("timeout_seconds", 15.0)),
        "max_retries": int(profile.get("max_retries", 2)),
        "extra_config": redact_secrets(dict(profile.get("extra_config") or {})),
        "is_demo": bool(profile.get("is_demo")),
        "source_updated_at": profile.get("updated_at"),
    }
    snap["snapshot_hash"] = sha256_hex(snap)
    return snap


class AuditWorkerService:
    """Single-worker audit orchestrator and lifecycle manager."""

    # shortcut: single-process background thread + SQLite atomic CAS lock; upgrade to OS-level file lock if running multi-process servers.
    def __init__(self, repo: Repository, start_background_loop: bool = True) -> None:
        self.repo = repo
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._active_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

        self.recover_unfinished_jobs_on_startup()

        if start_background_loop:
            self.start_worker_thread()

    def start_worker_thread(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="ModelTraceSingleWorker", daemon=True
        )
        self._worker_thread.start()

    def stop_worker_thread(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def recover_unfinished_jobs_on_startup(self) -> List[str]:
        """Mark any unfinished RUNNING audits as INTERRUPTED and IN_FLIGHT attempts as UNCERTAIN_PRIOR_ATTEMPT."""
        conn = self.repo._conn()
        interrupted_ids: List[str] = []
        now = utc_now_iso()
        try:
            with conn:
                rows = conn.execute(
                    "SELECT id FROM audits WHERE execution_state = ?;",
                    (ExecutionState.RUNNING.value,),
                ).fetchall()
                for r in rows:
                    aid = r["id"]
                    interrupted_ids.append(aid)
                    cur_att = conn.execute(
                        """
                        UPDATE sample_attempts
                        SET status = ?, error_code = 'PROCESS_RESTART_INTERRUPTED',
                            error_detail = 'Attempt was IN_FLIGHT when process restarted; external completion is uncertain.',
                            completed_at = ?
                        WHERE audit_id = ? AND status = ?;
                        """,
                        (
                            AttemptStatus.UNCERTAIN_PRIOR_ATTEMPT.value,
                            now,
                            aid,
                            AttemptStatus.IN_FLIGHT.value,
                        ),
                    )
                    uncertain_cnt = int(cur_att.rowcount or 0)
                    conn.execute(
                        """
                        UPDATE audits
                        SET execution_state = ?, verdict = ?,
                            verdict_reason = 'Audit was interrupted by process restart before completion. Paid external calls are never silently repeated; use explicit resume to continue.',
                            uncertain_prior_attempts = uncertain_prior_attempts + ?,
                            updated_at = ?
                        WHERE id = ?;
                        """,
                        (
                            ExecutionState.INTERRUPTED.value,
                            AuditVerdict.INCONCLUSIVE.value,
                            uncertain_cnt,
                            now,
                            aid,
                        ),
                    )
            for aid in interrupted_ids:
                self.repo.append_audit_event(
                    aid,
                    "AUDIT_INTERRUPTED_ON_RESTART",
                    "Unfinished RUNNING job marked INTERRUPTED on startup recovery. Completed samples are preserved; uncertain IN_FLIGHT attempts are recorded conservatively.",
                    level="WARNING",
                )
        finally:
            conn.close()
        return interrupted_ids

    # ------------------------------------------------------------------
    # Adapter instantiation & capability checks
    # ------------------------------------------------------------------
    def build_adapter(
        self,
        profile: Dict[str, Any],
        allowlist_snapshot: Optional[Dict[str, List[str]]] = None,
    ) -> Any:
        settings = allowlist_snapshot or self.repo.get_settings()
        atype = profile["adapter_type"]
        if atype == AdapterType.SYNTHETIC_DEMO.value:
            return SyntheticDemoAdapter(profile)
        if atype == AdapterType.OPENAI_COMPATIBLE.value:
            return OpenAICompatibleAdapter(
                profile=profile,
                allowed_outbound_origins=settings["allowed_outbound_origins"],
                allowed_local_inference_origins=settings["allowed_local_inference_origins"],
            )
        if atype == AdapterType.LOCAL_TRANSFORMERS.value:
            return LocalTransformersAdapter(profile=profile)
        raise ValueError(f"Unsupported adapter_type '{atype}'.")

    def test_profile_connection(
        self, profile_id: str, execute_network_probe: bool = True
    ) -> Dict[str, Any]:
        profile = self.repo.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile '{profile_id}' not found.")
        adapter = self.build_adapter(profile)
        try:
            if profile["adapter_type"] == AdapterType.OPENAI_COMPATIBLE.value:
                cap = adapter.check_capability(execute_network_probe=execute_network_probe)
            else:
                cap = adapter.check_capability()
        except AdapterError as exc:
            cap = {
                "status": "unavailable",
                "adapter_type": profile["adapter_type"],
                "is_demo": bool(profile.get("is_demo")),
                "reason_code": exc.error_code,
                "detail": redact_text(exc.message),
                "http_status": exc.http_status,
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }
        self.repo.record_profile_capability(profile_id, cap)
        return cap

    # ------------------------------------------------------------------
    # Audit Preview & Creation
    # ------------------------------------------------------------------
    def preview_audit(self, req: AuditPreviewRequest) -> Dict[str, Any]:
        ref_prof = self.repo.get_profile(req.reference_profile_id)
        if not ref_prof:
            raise ValueError(f"Reference profile '{req.reference_profile_id}' not found.")
        tgt_prof = self.repo.get_profile(req.target_profile_id)
        if not tgt_prof:
            raise ValueError(f"Target profile '{req.target_profile_id}' not found.")

        probe_rev = (
            self.repo.get_probe_revision(req.probe_revision_id)
            if req.probe_revision_id
            else self.repo.get_default_probe_revision()
        )
        if not probe_rev:
            raise ValueError(f"Probe revision '{req.probe_revision_id}' not found.")

        if req.n_prompts > probe_rev["prompt_count"]:
            raise ValueError(
                f"Requested n_prompts ({req.n_prompts}) exceeds available prompts in revision '{probe_rev['id']}' ({probe_rev['prompt_count']})."
            )

        scoring_prof = ref_prof
        if req.scoring_reference_profile_id:
            sp = self.repo.get_profile(req.scoring_reference_profile_id)
            if not sp:
                raise ValueError(
                    f"Scoring reference profile '{req.scoring_reference_profile_id}' not found."
                )
            scoring_prof = sp

        is_demo = bool(
            ref_prof.get("is_demo") or tgt_prof.get("is_demo") or scoring_prof.get("is_demo")
        )
        if is_demo and not (ref_prof.get("is_demo") and tgt_prof.get("is_demo")):
            raise ValueError(
                "Cannot mix DEMO synthetic profiles with live non-DEMO profiles in the same audit."
            )

        scoring_cap = self.build_adapter(scoring_prof).check_capability()
        warnings: List[str] = []
        blockers: List[str] = []

        if req.score_mode == ScoreMode.LOCAL_LOGIT_RANK:
            if not scoring_cap.get("supports_local_logit_scoring"):
                blockers.append(
                    f"Score mode 'local_logit_rank' requires local full-vocabulary logit scoring capability on reference profile '{scoring_prof['name']}', "
                    f"which reported status='{scoring_cap.get('status')}' ({scoring_cap.get('reason_code') or 'unsupported'}). "
                    "Configure an available local_transformers reference checkpoint or switch score_mode to 'length_distribution_only'."
                )
        else:
            warnings.append(
                "Response-only fallback mode ('length_distribution_only') evaluates Unicode response length only. "
                "It cannot verify token-level identity or inherit local logit-rank sensitivity."
            )

        for label, prof in [("reference", ref_prof), ("target", tgt_prof)]:
            cap = self.build_adapter(prof).check_capability()
            if cap.get("status") != "available":
                blockers.append(
                    f"{label.capitalize()} profile '{prof['name']}' is unavailable: {cap.get('reason_code')} — {cap.get('detail')}"
                )

        nominal_ref_calls = int(req.n_prompts) * int(req.m_reference)
        nominal_tgt_calls = int(req.n_prompts)
        nominal_total_calls = nominal_ref_calls + nominal_tgt_calls

        max_tgt_attempts = req.max_target_attempts or max(
            nominal_tgt_calls + min(nominal_tgt_calls, 10), nominal_tgt_calls
        )
        max_ref_attempts = req.max_reference_attempts or max(
            nominal_ref_calls + min(nominal_ref_calls // 2, 24), nominal_ref_calls
        )
        max_total_attempts = req.max_total_attempts or (max_tgt_attempts + max_ref_attempts)

        if max_tgt_attempts < nominal_tgt_calls:
            blockers.append(
                f"max_target_attempts ({max_tgt_attempts}) is less than nominal required target samples ({nominal_tgt_calls})."
            )
        if max_ref_attempts < nominal_ref_calls:
            blockers.append(
                f"max_reference_attempts ({max_ref_attempts}) is less than nominal required reference samples ({nominal_ref_calls})."
            )
        if max_total_attempts < nominal_total_calls:
            blockers.append(
                f"max_total_attempts ({max_total_attempts}) is less than nominal total samples ({nominal_total_calls})."
            )

        calibration = (
            self.repo.get_calibration(req.calibration_id) if req.calibration_id else None
        )
        if req.calibration_id and not calibration:
            blockers.append(f"Calibration artifact '{req.calibration_id}' not found.")

        ref_identity = build_reference_scorer_identity(ref_prof, scoring_prof)
        cal_check = check_calibration_compatibility(
            calibration=calibration,
            score_mode=req.score_mode.value,
            probe_revision_hash=probe_rev["content_hash"],
            n_prompts=req.n_prompts,
            m_reference=req.m_reference,
            alpha=req.alpha,
            audit_is_demo=is_demo,
            probe_revision_id=probe_rev["id"],
            temperature=req.temperature,
            top_p=req.top_p,
            max_output_tokens=req.max_output_tokens,
            reference_model_descriptor=ref_prof["model_identifier"],
            reference_identity=ref_identity,
        )
        if not cal_check["compatible"]:
            warnings.append(cal_check["reason"])

        # Honest cost estimation: Unknown cost is NEVER reported as $0.00
        settings = self.repo.get_settings()
        saved_pricing = settings.get("pricing_assumptions") or {}
        ref_rate = (
            req.reference_cost_per_1k_tokens_usd
            if req.reference_cost_per_1k_tokens_usd is not None
            else saved_pricing.get("reference_cost_per_1k_tokens_usd")
        )
        tgt_rate = (
            req.target_cost_per_1k_tokens_usd
            if req.target_cost_per_1k_tokens_usd is not None
            else saved_pricing.get("target_cost_per_1k_tokens_usd")
        )
        prompt_tok_est = int(saved_pricing.get("estimated_prompt_tokens_per_call", 64))
        tokens_per_call_bound = prompt_tok_est + int(req.max_output_tokens)

        if is_demo:
            cost_estimate = {
                "status": "DEMO_NO_EXTERNAL_BILLING",
                "is_known": False,
                "nominal_usd": None,
                "max_cap_usd": None,
                "display_label": "DEMO (Local synthetic simulation — no external API provider billing)",
                "assumptions": "Synthetic DEMO mode executes in-process with zero network calls.",
            }
        elif ref_rate is not None and tgt_rate is not None:
            nominal_usd = (
                (nominal_ref_calls * tokens_per_call_bound / 1000.0) * float(ref_rate)
                + (nominal_tgt_calls * tokens_per_call_bound / 1000.0) * float(tgt_rate)
            )
            max_cap_usd = (
                (max_ref_attempts * tokens_per_call_bound / 1000.0) * float(ref_rate)
                + (max_tgt_attempts * tokens_per_call_bound / 1000.0) * float(tgt_rate)
            )
            cost_estimate = {
                "status": "ESTIMATED_FROM_CONFIGURED_ASSUMPTIONS",
                "is_known": True,
                "nominal_usd": round(nominal_usd, 4),
                "max_cap_usd": round(max_cap_usd, 4),
                "reference_cost_per_1k_tokens_usd": float(ref_rate),
                "target_cost_per_1k_tokens_usd": float(tgt_rate),
                "estimated_tokens_per_call": tokens_per_call_bound,
                "display_label": f"Est. ${nominal_usd:.4f} nominal / ${max_cap_usd:.4f} cap bound",
                "assumptions": (
                    f"Assumes ~{prompt_tok_est} prompt tokens + up to {req.max_output_tokens} output tokens per call "
                    f"at ref=${float(ref_rate):.4f}/1k and target=${float(tgt_rate):.4f}/1k."
                ),
            }
        else:
            cost_estimate = {
                "status": "UNKNOWN_PRICING_NOT_CONFIGURED",
                "is_known": False,
                "nominal_usd": None,
                "max_cap_usd": None,
                "display_label": "UNKNOWN (Configure per-1k token pricing assumptions; unknown cost is not $0.00)",
                "assumptions": "No provider token pricing rates configured in Settings or request.",
            }

        seeds = derive_audit_seeds(req.master_seed)

        return {
            "ready_to_start": len(blockers) == 0,
            "is_demo": is_demo,
            "score_mode": req.score_mode.value,
            "score_scope_banner": (
                "LENGTH DISTRIBUTION ONLY — Cannot verify exact model identity or token probabilities."
                if req.score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY
                else "LOCAL_LOGIT_RANK — Teacher-forcing mean log vocabulary rank under reference logits."
            ),
            "probe_revision": {
                "id": probe_rev["id"],
                "name": probe_rev["name"],
                "revision_number": probe_rev["revision_number"],
                "content_hash": probe_rev["content_hash"],
                "validation_label": probe_rev["validation_label"],
            },
            "budget": {
                "n_prompts": req.n_prompts,
                "m_reference_per_prompt": req.m_reference,
                "nominal_reference_calls": nominal_ref_calls,
                "nominal_target_calls": nominal_tgt_calls,
                "nominal_total_calls": nominal_total_calls,
                "max_output_tokens": req.max_output_tokens,
                "max_target_attempts": max_tgt_attempts,
                "max_reference_attempts": max_ref_attempts,
                "max_total_attempts": max_total_attempts,
                "max_duration_seconds": req.max_duration_seconds,
                "estimated_max_output_tokens_total": nominal_total_calls * req.max_output_tokens,
                "cost_estimate": cost_estimate,
            },
            "statistical_plan": {
                "method": STAT_ENGINE_METHOD,
                "version": STAT_ENGINE_VERSION,
                "alpha": req.alpha,
                "mc_simulations": req.mc_simulations,
                "derived_seeds": seeds,
                "screening_budget_disclaimer": (
                    "Initial screening defaults (n=20, m=8, max_tokens<=128, alpha=0.01) represent a "
                    "bounded operational budget, not a universal detection guarantee."
                ),
            },
            "calibration_compatibility": cal_check,
            "warnings": warnings,
            "blockers": blockers,
        }

    def create_audit(self, req: AuditCreateRequest) -> Dict[str, Any]:
        """Freeze audit manifest and persist audit record in CREATED state (or start if requested)."""
        preview = self.preview_audit(req)
        if not preview["ready_to_start"]:
            raise ValueError(
                "Audit configuration failed pre-flight validation: "
                + "; ".join(preview["blockers"])
            )

        ref_prof = self.repo.get_profile(req.reference_profile_id)
        tgt_prof = self.repo.get_profile(req.target_profile_id)
        scoring_prof_id = req.scoring_reference_profile_id or req.reference_profile_id
        scoring_prof = self.repo.get_profile(scoring_prof_id)
        assert ref_prof is not None and tgt_prof is not None and scoring_prof is not None

        probe_rev = (
            self.repo.get_probe_revision(req.probe_revision_id)
            if req.probe_revision_id
            else self.repo.get_default_probe_revision()
        )
        assert probe_rev is not None

        selected_prompts = [dict(p) for p in probe_rev["prompts"][: req.n_prompts]]
        seeds = preview["statistical_plan"]["derived_seeds"]
        budget = preview["budget"]
        is_demo = bool(preview["is_demo"])

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        audit_name = req.name
        if is_demo and not audit_name.startswith("[DEMO]"):
            audit_name = f"[DEMO] {audit_name}"

        frozen_ref = freeze_profile_snapshot(ref_prof)
        frozen_tgt = freeze_profile_snapshot(tgt_prof)
        frozen_scorer = freeze_profile_snapshot(scoring_prof)
        ref_identity = build_reference_scorer_identity(frozen_ref, frozen_scorer)
        cur_settings = self.repo.get_settings()

        manifest: Dict[str, Any] = {
            "audit_id": audit_id,
            "name": audit_name,
            "is_demo": is_demo,
            "frozen_at": now,
            "reference_profile": frozen_ref,
            "target_profile": frozen_tgt,
            "scoring_profile": frozen_scorer,
            "reference_identity": ref_identity,
            "scoring_reference_profile_id": scoring_prof_id,
            "probe_revision_id": probe_rev["id"],
            "probe_revision_hash": probe_rev["content_hash"],
            "selected_prompt_ids": [p["prompt_id"] for p in selected_prompts],
            "selected_prompts_snapshot": selected_prompts,
            "network_allowlist_snapshot": {
                "allowed_outbound_origins": list(cur_settings["allowed_outbound_origins"]),
                "allowed_local_inference_origins": list(cur_settings["allowed_local_inference_origins"]),
            },
            "calibration_id": req.calibration_id,
            "score_mode": req.score_mode.value,
            "sampling": {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_output_tokens": req.max_output_tokens,
            },
            "statistics": {
                "method": STAT_ENGINE_METHOD,
                "version": STAT_ENGINE_VERSION,
                "n_prompts": req.n_prompts,
                "m_reference": req.m_reference,
                "alpha": req.alpha,
                "mc_simulations": req.mc_simulations,
                "include_secondary_tests": req.include_secondary_tests,
                "seeds": seeds,
            },
            "caps": {
                "max_target_attempts": budget["max_target_attempts"],
                "max_reference_attempts": budget["max_reference_attempts"],
                "max_total_attempts": budget["max_total_attempts"],
                "max_duration_seconds": budget["max_duration_seconds"],
                "cost_estimate": budget["cost_estimate"],
            },
            "epistemic_disclaimer": (
                "Evidence concerns observable output behavior under this frozen configuration, "
                "not cryptographic proof of model weights, parameter count, precision, or provider intent."
            ),
        }
        manifest_canonical = canonical_json(manifest)
        manifest_hash = sha256_hex(manifest_canonical)

        conn = self.repo._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO audits (
                        id, name, notes, reference_profile_id, target_profile_id,
                        scoring_reference_profile_id, probe_revision_id, probe_revision_hash,
                        calibration_id, score_mode, is_demo, n_prompts, m_reference,
                        max_output_tokens, alpha, temperature, top_p, mc_simulations,
                        include_secondary_tests, max_target_attempts, max_reference_attempts,
                        max_total_attempts, max_duration_seconds, master_seed, seeds_json,
                        manifest_json, manifest_hash, execution_state, verdict, verdict_reason,
                        cancel_requested, resume_count, uncertain_prior_attempts,
                        local_scoring_compute_ms, summary_json, integrity_hash,
                        created_at, started_at, completed_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0.0, NULL, NULL,
                        ?, NULL, NULL, ?
                    );
                    """,
                    (
                        audit_id,
                        audit_name,
                        req.notes,
                        req.reference_profile_id,
                        req.target_profile_id,
                        req.scoring_reference_profile_id or req.reference_profile_id,
                        probe_rev["id"],
                        probe_rev["content_hash"],
                        req.calibration_id,
                        req.score_mode.value,
                        1 if is_demo else 0,
                        req.n_prompts,
                        req.m_reference,
                        req.max_output_tokens,
                        req.alpha,
                        req.temperature,
                        req.top_p,
                        req.mc_simulations,
                        1 if req.include_secondary_tests else 0,
                        budget["max_target_attempts"],
                        budget["max_reference_attempts"],
                        budget["max_total_attempts"],
                        budget["max_duration_seconds"],
                        req.master_seed,
                        canonical_json(seeds),
                        manifest_canonical,
                        manifest_hash,
                        ExecutionState.CREATED.value,
                        AuditVerdict.INCONCLUSIVE.value,
                        "Audit manifest frozen in CREATED state; awaiting deliberate start.",
                        now,
                        now,
                    ),
                )
        finally:
            conn.close()

        self.repo.append_audit_event(
            audit_id,
            "AUDIT_CREATED",
            f"Frozen audit manifest (hash {manifest_hash[:12]}...) with n={req.n_prompts}, m={req.m_reference}, score_mode={req.score_mode.value}.",
            details={"manifest_hash": manifest_hash, "is_demo": is_demo},
        )

        if req.start_immediately:
            return self.start_audit(audit_id)
        return self.repo.get_audit(audit_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Start / Cancel / Explicit Resume
    # ------------------------------------------------------------------
    def start_audit(self, audit_id: str, run_synchronously: bool = False) -> Dict[str, Any]:
        """Atomically transition a CREATED audit to RUNNING, preventing double starts."""
        now = utc_now_iso()
        conn = self.repo._conn()
        try:
            with conn:
                row = conn.execute(
                    "SELECT execution_state FROM audits WHERE id = ?;", (audit_id,)
                ).fetchone()
                if not row:
                    raise KeyError(f"Audit '{audit_id}' not found.")
                cur_state = row["execution_state"]
                if cur_state != ExecutionState.CREATED.value:
                    raise RuntimeError(
                        f"Cannot start audit '{audit_id}' because its execution_state is '{cur_state}' (double start prevented)."
                    )
                cur = conn.execute(
                    """
                    UPDATE audits
                    SET execution_state = ?, started_at = COALESCE(started_at, ?),
                        cancel_requested = 0, updated_at = ?
                    WHERE id = ? AND execution_state = ?;
                    """,
                    (
                        ExecutionState.RUNNING.value,
                        now,
                        now,
                        audit_id,
                        ExecutionState.CREATED.value,
                    ),
                )
                if cur.rowcount == 0:
                    raise RuntimeError(f"Concurrent start detected for audit '{audit_id}'.")
        finally:
            conn.close()

        self.repo.append_audit_event(
            audit_id,
            "AUDIT_STARTED",
            "Deliberate audit start initiated by operator.",
        )

        if run_synchronously:
            self.execute_audit_job(audit_id)
        else:
            self._wake_event.set()
        return self.repo.get_audit(audit_id)  # type: ignore[return-value]

    def cancel_audit(self, audit_id: str) -> Dict[str, Any]:
        """Request cooperative cancellation of a CREATED or RUNNING audit."""
        now = utc_now_iso()
        conn = self.repo._conn()
        try:
            with conn:
                row = conn.execute(
                    "SELECT execution_state FROM audits WHERE id = ?;", (audit_id,)
                ).fetchone()
                if not row:
                    raise KeyError(f"Audit '{audit_id}' not found.")
                state = row["execution_state"]
                if state == ExecutionState.CREATED.value:
                    conn.execute(
                        """
                        UPDATE audits
                        SET execution_state = ?, verdict = ?,
                            verdict_reason = 'Audit cancelled before execution began.',
                            cancel_requested = 1, completed_at = ?, updated_at = ?
                        WHERE id = ?;
                        """,
                        (
                            ExecutionState.CANCELLED.value,
                            AuditVerdict.INCONCLUSIVE.value,
                            now,
                            now,
                            audit_id,
                        ),
                    )
                elif state == ExecutionState.RUNNING.value:
                    conn.execute(
                        "UPDATE audits SET cancel_requested = 1, updated_at = ? WHERE id = ?;",
                        (now, audit_id),
                    )
                else:
                    raise RuntimeError(
                        f"Audit '{audit_id}' is in terminal/non-running state '{state}' and cannot be cancelled."
                    )
        finally:
            conn.close()

        self.repo.append_audit_event(
            audit_id,
            "AUDIT_CANCEL_REQUESTED",
            "Operator requested audit cancellation.",
            level="WARNING",
        )
        return self.repo.get_audit(audit_id)  # type: ignore[return-value]

    def resume_audit(self, audit_id: str, run_synchronously: bool = False) -> Dict[str, Any]:
        """Explicitly resume an INTERRUPTED audit, reusing completed samples and accounting conservatively for uncertain calls."""
        now = utc_now_iso()
        conn = self.repo._conn()
        try:
            with conn:
                row = conn.execute(
                    "SELECT execution_state, resume_count, uncertain_prior_attempts FROM audits WHERE id = ?;",
                    (audit_id,),
                ).fetchone()
                if not row:
                    raise KeyError(f"Audit '{audit_id}' not found.")
                if row["execution_state"] != ExecutionState.INTERRUPTED.value:
                    raise RuntimeError(
                        f"Only INTERRUPTED audits can be resumed (current state: '{row['execution_state']}')."
                    )
                cur = conn.execute(
                    """
                    UPDATE audits
                    SET execution_state = ?, resume_count = resume_count + 1,
                        cancel_requested = 0, updated_at = ?
                    WHERE id = ? AND execution_state = ?;
                    """,
                    (
                        ExecutionState.RUNNING.value,
                        now,
                        audit_id,
                        ExecutionState.INTERRUPTED.value,
                    ),
                )
                if cur.rowcount == 0:
                    raise RuntimeError(f"Concurrent resume detected for audit '{audit_id}'.")
        finally:
            conn.close()

        counts = self.repo.get_attempt_counts(audit_id)
        self.repo.append_audit_event(
            audit_id,
            "AUDIT_RESUMED",
            (
                f"Explicit resume initiated. Reusing {counts['completed_reference_samples']} reference and "
                f"{counts['completed_target_samples']} target completed responses. "
                f"{counts['uncertain_prior_attempts']} uncertain prior IN_FLIGHT attempt(s) remain counted "
                "against total attempt caps (exactly-once external calls cannot be promised)."
            ),
            details=counts,
        )

        if run_synchronously:
            self.execute_audit_job(audit_id)
        else:
            self._wake_event.set()
        return self.repo.get_audit(audit_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Background Worker Loop & Core Job Execution
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            ran_any = False
            conn = self.repo._conn()
            try:
                row = conn.execute(
                    "SELECT id FROM audits WHERE execution_state = ? ORDER BY updated_at ASC LIMIT 1;",
                    (ExecutionState.RUNNING.value,),
                ).fetchone()
            finally:
                conn.close()

            if row:
                ran_any = True
                try:
                    self.execute_audit_job(row["id"])
                except Exception as exc:  # pragma: no cover
                    self._fail_audit(
                        row["id"],
                        "UNHANDLED_WORKER_EXCEPTION",
                        f"Worker encountered unexpected error: {redact_text(str(exc))}",
                    )

            if not ran_any:
                self._wake_event.wait(timeout=0.25)
                self._wake_event.clear()

    def _is_cancel_requested(self, audit_id: str) -> bool:
        conn = self.repo._conn()
        try:
            row = conn.execute(
                "SELECT cancel_requested, execution_state FROM audits WHERE id = ?;",
                (audit_id,),
            ).fetchone()
            if not row:
                return True
            return bool(row["cancel_requested"]) or (
                row["execution_state"] != ExecutionState.RUNNING.value
            )
        finally:
            conn.close()

    def _fail_audit(self, audit_id: str, error_code: str, reason: str) -> None:
        now = utc_now_iso()
        clean_reason = redact_text(reason)
        conn = self.repo._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE audits
                    SET execution_state = ?, verdict = ?, verdict_reason = ?,
                        completed_at = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (
                        ExecutionState.FAILED.value,
                        AuditVerdict.INCONCLUSIVE.value,
                        f"[{error_code}] {clean_reason}",
                        now,
                        now,
                        audit_id,
                    ),
                )
        finally:
            conn.close()
        self.repo.append_audit_event(
            audit_id,
            "AUDIT_FAILED",
            f"Audit failed ({error_code}): {clean_reason}",
            level="ERROR",
            details={"error_code": error_code},
        )

    def _finalize_cancelled_audit(self, audit_id: str) -> None:
        now = utc_now_iso()
        conn = self.repo._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE audits
                    SET execution_state = ?, verdict = ?,
                        verdict_reason = 'Audit cancelled by operator before completion; incomplete samples cannot yield a valid verdict.',
                        completed_at = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (
                        ExecutionState.CANCELLED.value,
                        AuditVerdict.INCONCLUSIVE.value,
                        now,
                        now,
                        audit_id,
                    ),
                )
        finally:
            conn.close()
        self.repo.append_audit_event(
            audit_id,
            "AUDIT_CANCELLED",
            "Audit execution stopped cleanly after cancellation request.",
            level="WARNING",
        )

    def execute_audit_job(self, audit_id: str) -> None:
        """Execute or resume a single RUNNING audit job to completion, failure, or cancellation."""
        with self._active_lock:
            audit = self.repo.get_audit(audit_id)
            if not audit or audit["execution_state"] != ExecutionState.RUNNING.value:
                return

            job_t0 = time.monotonic()
            max_duration = float(audit["max_duration_seconds"])
            max_ref_attempts = int(audit["max_reference_attempts"])
            max_tgt_attempts = int(audit["max_target_attempts"])
            max_tot_attempts = int(audit["max_total_attempts"])

            manifest = audit.get("manifest") or {}
            ref_prof = manifest.get("reference_profile")
            tgt_prof = manifest.get("target_profile")
            scoring_prof = manifest.get("scoring_profile") or ref_prof
            selected_prompts = manifest.get("selected_prompts_snapshot")

            if not selected_prompts:
                probe_rev = self.repo.get_probe_revision(audit["probe_revision_id"])
                if probe_rev:
                    selected_prompts = probe_rev["prompts"][: int(audit["n_prompts"])]

            if not (
                isinstance(ref_prof, dict)
                and ref_prof.get("adapter_type")
                and isinstance(tgt_prof, dict)
                and tgt_prof.get("adapter_type")
                and isinstance(scoring_prof, dict)
                and scoring_prof.get("adapter_type")
                and selected_prompts
            ):
                self._fail_audit(
                    audit_id,
                    "MISSING_FROZEN_MANIFEST_SNAPSHOT",
                    "Frozen reference profile, target profile, scorer snapshot, or probe snapshot missing.",
                )
                return

            allowlist_snap = manifest.get("network_allowlist_snapshot")
            ref_adapter = self.build_adapter(ref_prof, allowlist_snapshot=allowlist_snap)
            tgt_adapter = self.build_adapter(tgt_prof, allowlist_snapshot=allowlist_snap)
            scoring_adapter = self.build_adapter(scoring_prof, allowlist_snapshot=allowlist_snap)

            score_mode = str(audit["score_mode"])
            n_prompts = int(audit["n_prompts"])
            m_ref = int(audit["m_reference"])
            temperature = float(audit["temperature"])
            top_p = float(audit["top_p"])
            max_tokens = int(audit["max_output_tokens"])
            seeds = audit["seeds"]

            existing_resps = self.repo.list_audit_responses(audit_id)
            completed_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
            for r in existing_resps:
                completed_map[(r["prompt_id"], r["role"], int(r["sample_index"]))] = r

            for p_idx, prompt_obj in enumerate(selected_prompts):
                prompt_id = str(prompt_obj["prompt_id"])
                prompt_text = str(prompt_obj["prompt_text"])

                sample_tasks: List[Tuple[str, int, Dict[str, Any], Any, int]] = []
                for j in range(m_ref):
                    step_seed = int((seeds["reference_seed"] + p_idx * 1009 + j * 37) & 0x7FFFFFFF)
                    sample_tasks.append(("reference", j, ref_prof, ref_adapter, step_seed))
                tgt_step_seed = int((seeds["target_seed"] + p_idx * 1009 + 997) & 0x7FFFFFFF)
                sample_tasks.append(("target", 0, tgt_prof, tgt_adapter, tgt_step_seed))

                for role, s_idx, prof, adapter, step_seed in sample_tasks:
                    if (prompt_id, role, s_idx) in completed_map:
                        continue

                    if self._is_cancel_requested(audit_id):
                        self._finalize_cancelled_audit(audit_id)
                        return

                    elapsed_sec = time.monotonic() - job_t0
                    if elapsed_sec > max_duration:
                        self._fail_audit(
                            audit_id,
                            "MAX_DURATION_EXCEEDED",
                            f"Audit exceeded max_duration_seconds ({max_duration}s).",
                        )
                        return

                    counts = self.repo.get_attempt_counts(audit_id)
                    if counts["total_attempts_used"] >= max_tot_attempts:
                        self._fail_audit(
                            audit_id,
                            "MAX_TOTAL_ATTEMPTS_EXCEEDED",
                            f"Total attempt cap ({max_tot_attempts}) reached (including retries/uncertain calls).",
                        )
                        return
                    if role == "reference" and counts["reference_attempts_used"] >= max_ref_attempts:
                        self._fail_audit(
                            audit_id,
                            "MAX_REFERENCE_ATTEMPTS_EXCEEDED",
                            f"Reference attempt cap ({max_ref_attempts}) reached.",
                        )
                        return
                    if role == "target" and counts["target_attempts_used"] >= max_tgt_attempts:
                        self._fail_audit(
                            audit_id,
                            "MAX_TARGET_ATTEMPTS_EXCEEDED",
                            f"Target attempt cap ({max_tgt_attempts}) reached.",
                        )
                        return

                    attempt_num = counts["total_attempts_used"] + 1
                    attempt_id = self.repo.create_sample_attempt(
                        audit_id=audit_id,
                        prompt_id=prompt_id,
                        prompt_index=p_idx,
                        role=role,
                        sample_index=s_idx,
                        attempt_number=attempt_num,
                        profile_id=prof["id"],
                    )

                    active_attempt_Holder = {"id": attempt_id}

                    def _on_retry_hook(retry_idx: int) -> None:
                        self.repo.fail_sample_attempt(
                            active_attempt_Holder["id"],
                            error_code="TRANSIENT_RETRY",
                            error_detail=f"Transient error on attempt; scheduling retry #{retry_idx}.",
                            retry_count=retry_idx - 1,
                        )
                        cur_c = self.repo.get_attempt_counts(audit_id)
                        if cur_c["total_attempts_used"] >= max_tot_attempts:
                            raise AdapterError(
                                "MAX_TOTAL_ATTEMPTS_EXCEEDED",
                                f"Retry blocked: total attempt cap ({max_tot_attempts}) reached.",
                                retryable=False,
                            )
                        if (
                            role == "reference"
                            and cur_c["reference_attempts_used"] >= max_ref_attempts
                        ):
                            raise AdapterError(
                                "MAX_REFERENCE_ATTEMPTS_EXCEEDED",
                                f"Retry blocked: reference attempt cap ({max_ref_attempts}) reached.",
                                retryable=False,
                            )
                        if role == "target" and cur_c["target_attempts_used"] >= max_tgt_attempts:
                            raise AdapterError(
                                "MAX_TARGET_ATTEMPTS_EXCEEDED",
                                f"Retry blocked: target attempt cap ({max_tgt_attempts}) reached.",
                                retryable=False,
                            )
                        new_att_id = self.repo.create_sample_attempt(
                            audit_id=audit_id,
                            prompt_id=prompt_id,
                            prompt_index=p_idx,
                            role=role,
                            sample_index=s_idx,
                            attempt_number=cur_c["total_attempts_used"] + 1,
                            profile_id=prof["id"],
                        )
                        active_attempt_Holder["id"] = new_att_id

                    try:
                        step_rng = np.random.default_rng(step_seed)
                        if prof["adapter_type"] == AdapterType.SYNTHETIC_DEMO.value:
                            gen_res: GenerationResult = adapter.generate(
                                prompt_id=prompt_id,
                                prompt_text=prompt_text,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                rng=step_rng,
                            )
                        elif prof["adapter_type"] == AdapterType.OPENAI_COMPATIBLE.value:
                            gen_res = adapter.generate(
                                prompt_id=prompt_id,
                                prompt_text=prompt_text,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                seed=step_seed,
                                on_attempt_hook=_on_retry_hook,
                            )
                        else:
                            gen_res = adapter.generate(
                                prompt_id=prompt_id,
                                prompt_text=prompt_text,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                seed=step_seed,
                            )

                        if score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value:
                            t_sc = time.perf_counter()
                            raw_score = compute_unicode_length_score(gen_res.response_text)
                            norm_ll = None
                            sc_ms = (time.perf_counter() - t_sc) * 1000.0
                        else:
                            if scoring_prof["adapter_type"] == AdapterType.SYNTHETIC_DEMO.value:
                                sc_res: ScoringResult = scoring_adapter.score_response(
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_text,
                                    response_text=gen_res.response_text,
                                    observable_meta=gen_res.observable_meta,
                                    score_mode=score_mode,
                                )
                            else:
                                sc_res = scoring_adapter.score_response(
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_text,
                                    response_text=gen_res.response_text,
                                )
                            raw_score = sc_res.raw_score
                            norm_ll = sc_res.normalized_log_likelihood
                            sc_ms = sc_res.scoring_compute_ms

                        unicode_len = int(compute_unicode_length_score(gen_res.response_text))
                        self.repo.complete_sample_attempt(
                            attempt_id=active_attempt_Holder["id"],
                            audit_id=audit_id,
                            prompt_id=prompt_id,
                            role=role,
                            sample_index=s_idx,
                            response_text=gen_res.response_text,
                            token_count=gen_res.token_count,
                            unicode_length=unicode_len,
                            finish_reason=gen_res.finish_reason,
                            declared_model=gen_res.declared_model,
                            observable_meta=gen_res.observable_meta,
                            raw_score=raw_score,
                            normalized_log_likelihood=norm_ll,
                            scoring_compute_ms=sc_ms,
                            latency_ms=gen_res.latency_ms,
                            http_status=gen_res.http_status,
                            retry_count=gen_res.retry_count,
                        )
                        completed_map[(prompt_id, role, s_idx)] = {
                            "raw_score": raw_score,
                            "normalized_log_likelihood": norm_ll,
                            "scoring_compute_ms": sc_ms,
                            "unicode_length": unicode_len,
                            "token_count": gen_res.token_count,
                            "latency_ms": gen_res.latency_ms,
                        }
                    except AdapterError as exc:
                        self.repo.fail_sample_attempt(
                            active_attempt_Holder["id"],
                            error_code=exc.error_code,
                            error_detail=exc.message,
                            http_status=exc.http_status,
                            latency_ms=exc.latency_ms,
                            retry_count=exc.retry_count,
                        )
                        self._fail_audit(
                            audit_id,
                            exc.error_code,
                            f"Sample failed on prompt '{prompt_id}' ({role} #{s_idx}): {exc.message}",
                        )
                        return
                    except Exception as exc:
                        self.repo.fail_sample_attempt(
                            active_attempt_Holder["id"],
                            error_code="ADAPTER_EXECUTION_ERROR",
                            error_detail=str(exc),
                        )
                        self._fail_audit(
                            audit_id,
                            "ADAPTER_EXECUTION_ERROR",
                            f"Unexpected error on prompt '{prompt_id}' ({role} #{s_idx}): {exc}",
                        )
                        return

            if self._is_cancel_requested(audit_id):
                self._finalize_cancelled_audit(audit_id)
                return

            self._compute_and_persist_final_statistics(audit_id, selected_prompts)

    def _compute_and_persist_final_statistics(
        self, audit_id: str, selected_prompts: List[Dict[str, Any]]
    ) -> None:
        audit = self.repo.get_audit(audit_id)
        assert audit is not None
        responses = self.repo.list_audit_responses(audit_id)
        by_prompt_role: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in responses:
            by_prompt_role.setdefault((r["prompt_id"], r["role"]), []).append(r)

        n_prompts = int(audit["n_prompts"])
        m_ref = int(audit["m_reference"])
        score_mode = str(audit["score_mode"])
        seeds = audit["seeds"]
        tie_rng = np.random.default_rng(int(seeds["tie_break_seed"]))

        prompt_score_records: List[Dict[str, Any]] = []
        z_values: List[float] = []
        all_complete = True

        for p_idx, prompt_obj in enumerate(selected_prompts):
            pid = str(prompt_obj["prompt_id"])
            fam = str(prompt_obj.get("family", "general"))
            ref_list = sorted(
                by_prompt_role.get((pid, "reference"), []),
                key=lambda x: int(x["sample_index"]),
            )
            tgt_list = by_prompt_role.get((pid, "target"), [])
            if len(ref_list) < m_ref or len(tgt_list) < 1:
                all_complete = False
                break

            ref_scores = [float(x["raw_score"]) for x in ref_list[:m_ref]]
            tgt_score = float(tgt_list[0]["raw_score"])

            rank_info = compute_prompt_rank_z(
                reference_scores=ref_scores,
                target_score=tgt_score,
                score_mode=score_mode,
                rng=tie_rng,
            )

            ref_lls = [
                float(x["normalized_log_likelihood"])
                for x in ref_list[:m_ref]
                if x["normalized_log_likelihood"] is not None
            ]
            ref_mean_ll = float(np.mean(ref_lls)) if ref_lls else None
            tgt_ll = tgt_list[0]["normalized_log_likelihood"]
            compute_ms = sum(float(x["scoring_compute_ms"]) for x in ref_list[:m_ref]) + float(
                tgt_list[0]["scoring_compute_ms"]
            )

            rec = {
                "id": f"psc_{audit_id[-6:]}_{p_idx:03d}",
                "audit_id": audit_id,
                "prompt_id": pid,
                "prompt_index": p_idx,
                "family": fam,
                "score_mode": score_mode,
                "m_reference": m_ref,
                "reference_scores": rank_info["quantized_reference_scores"],
                "target_score": rank_info["quantized_target_score"],
                "a_below": rank_info["a_below"],
                "b_ties": rank_info["b_ties"],
                "j_draw": rank_info["j_draw"],
                "u_draw": rank_info["u_draw"],
                "z_uniform": rank_info["z_uniform"],
                "score_shift": rank_info["score_shift"],
                "target_norm_ll": tgt_ll,
                "ref_mean_norm_ll": ref_mean_ll,
                "local_scoring_compute_ms": compute_ms,
            }
            self.repo.save_prompt_score(rec)
            prompt_score_records.append(rec)
            z_values.append(rank_info["z_uniform"])

        if not all_complete or len(prompt_score_records) < n_prompts:
            self._fail_audit(
                audit_id,
                "INCOMPLETE_SAMPLE_SET",
                f"Expected {n_prompts} complete prompts with {m_ref} references + 1 target each; got {len(prompt_score_records)}.",
            )
            return

        cvm_res = run_cvm_monte_carlo_test(
            z_values=z_values,
            mc_null_seed=int(seeds["mc_null_seed"]),
            n_simulations=int(audit["mc_simulations"]),
        )

        bootstrap_res = compute_bootstrap_uncertainty(
            prompt_records=prompt_score_records,
            bootstrap_seed=int(seeds["bootstrap_seed"]),
            n_replicates=1000,
        )

        secondary_inferential: List[Dict[str, Any]] = []
        if audit.get("include_secondary_tests"):
            shifts_arr = np.array([r["score_shift"] for r in prompt_score_records], dtype=np.float64)
            if np.all(shifts_arr == 0.0):
                wilcoxon_p = 1.0
            else:
                try:
                    w_res = scipy_stats.wilcoxon(shifts_arr, zero_method="zsplit")
                    wilcoxon_p = float(w_res.pvalue) if not np.isnan(w_res.pvalue) else 1.0
                except Exception:
                    wilcoxon_p = 1.0
            ks_res = scipy_stats.kstest(z_values, "uniform")
            raw_tests = [
                {
                    "name": "primary_cvm_uniform_rank",
                    "raw_p_value": cvm_res["p_value"],
                    "statistic": cvm_res["cvm_statistic"],
                },
                {
                    "name": "secondary_wilcoxon_score_shift",
                    "raw_p_value": wilcoxon_p,
                    "statistic": float(np.median(shifts_arr)),
                },
                {
                    "name": "secondary_ks_uniform_rank",
                    "raw_p_value": float(ks_res.pvalue),
                    "statistic": float(ks_res.statistic),
                },
            ]
            secondary_inferential = holm_step_down(raw_tests, alpha=float(audit["alpha"]))

        ref_lengths = [int(r["unicode_length"]) for r in responses if r["role"] == "reference"]
        tgt_lengths = [int(r["unicode_length"]) for r in responses if r["role"] == "target"]
        declared_models = sorted(
            {str(r["declared_model"]) for r in responses if r.get("declared_model")}
        )

        descriptive_diagnostics = {
            "scope": "DESCRIPTIVE_ONLY",
            "disclaimer": "Descriptive observations (latency, length, declared headers, normalized log-likelihood) do not establish model identity.",
            "mean_reference_unicode_length": float(np.mean(ref_lengths)) if ref_lengths else 0.0,
            "mean_target_unicode_length": float(np.mean(tgt_lengths)) if tgt_lengths else 0.0,
            "observed_declared_models_untrusted": declared_models,
            "reference_sample_count": len(ref_lengths),
            "target_sample_count": len(tgt_lengths),
        }

        calibration = (
            self.repo.get_calibration(audit["calibration_id"])
            if audit.get("calibration_id")
            else None
        )
        manifest = audit.get("manifest") or {}
        frozen_ref = manifest.get("reference_profile") or {}
        frozen_scorer = manifest.get("scoring_profile") or frozen_ref
        ref_identity = manifest.get("reference_identity") or build_reference_scorer_identity(
            frozen_ref, frozen_scorer
        )
        cal_check = check_calibration_compatibility(
            calibration=calibration,
            score_mode=score_mode,
            probe_revision_hash=audit["probe_revision_hash"],
            n_prompts=n_prompts,
            m_reference=m_ref,
            alpha=float(audit["alpha"]),
            audit_is_demo=bool(audit["is_demo"]),
            probe_revision_id=audit["probe_revision_id"],
            temperature=float(audit["temperature"]),
            top_p=float(audit["top_p"]),
            max_output_tokens=int(audit["max_output_tokens"]),
            reference_model_descriptor=str(frozen_ref.get("model_identifier") or ""),
            reference_identity=ref_identity,
        )

        verdict, verdict_reason = evaluate_verdict(
            execution_state=ExecutionState.COMPLETED.value,
            completed_prompts=len(prompt_score_records),
            expected_n_prompts=n_prompts,
            all_prompts_have_full_samples=True,
            p_value=cvm_res["p_value"],
            alpha=float(audit["alpha"]),
            score_mode=score_mode,
            calibration_check=cal_check,
            is_demo=bool(audit["is_demo"]),
        )

        attempt_counts = self.repo.get_attempt_counts(audit_id)
        updated_audit = self.repo.get_audit(audit_id)
        total_local_compute_ms = float(
            updated_audit["local_scoring_compute_ms"] if updated_audit else 0.0
        )

        summary: Dict[str, Any] = {
            "is_demo": bool(audit["is_demo"]),
            "score_mode": score_mode,
            "score_scope_label": (
                "LENGTH DISTRIBUTION ONLY"
                if score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value
                else "LOCAL_LOGIT_RANK"
            ),
            "execution_state": ExecutionState.COMPLETED.value,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "primary_test": cvm_res,
            "uncertainty": bootstrap_res,
            "calibration_evaluation": cal_check,
            "holm_corrected_tests": secondary_inferential,
            "descriptive_diagnostics": descriptive_diagnostics,
            "attempt_accounting": attempt_counts,
            "local_scoring_compute_ms": round(total_local_compute_ms, 3),
            "null_and_confounder_notes": [
                "Null hypothesis H0 assumes independent exchangeability of target and reference scores per prompt under identical nominal sampling settings.",
                "Confounders including chat template differences, hidden system prompts, moderation filters, stop token handling, or hardware kernel non-determinism can produce INCONSISTENT verdicts even when base weights match.",
                "Evidence hashes are SHA-256 integrity checks over stored records, not third-party cryptographic attestation.",
            ],
        }

        integrity_payload = {
            "manifest_hash": audit["manifest_hash"],
            "execution_state": ExecutionState.COMPLETED.value,
            "verdict": verdict,
            "primary_test": cvm_res,
            "z_values": [round(z, 9) for z in z_values],
        }
        integrity_hash = sha256_hex(integrity_payload)
        now = utc_now_iso()

        conn = self.repo._conn()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE audits
                    SET execution_state = ?, verdict = ?, verdict_reason = ?,
                        summary_json = ?, integrity_hash = ?, completed_at = ?, updated_at = ?
                    WHERE id = ?;
                    """,
                    (
                        ExecutionState.COMPLETED.value,
                        verdict,
                        verdict_reason,
                        canonical_json(summary),
                        integrity_hash,
                        now,
                        now,
                        audit_id,
                    ),
                )
        finally:
            conn.close()

        self.repo.append_audit_event(
            audit_id,
            "AUDIT_COMPLETED",
            f"Audit completed with verdict={verdict} (CvM W^2={cvm_res['cvm_statistic']:.4f}, p={cvm_res['p_value']:.5f}).",
            details={
                "verdict": verdict,
                "cvm_statistic": cvm_res["cvm_statistic"],
                "p_value": cvm_res["p_value"],
                "integrity_hash": integrity_hash,
            },
        )

    # ------------------------------------------------------------------
    # Progress Polling & Demo Power Calibration Generator
    # ------------------------------------------------------------------
    def get_audit_progress(self, audit_id: str) -> Dict[str, Any]:
        audit = self.repo.get_audit(audit_id)
        if not audit:
            raise KeyError(f"Audit '{audit_id}' not found.")
        counts = self.repo.get_attempt_counts(audit_id)
        m_ref = int(audit["m_reference"])
        n_prompts = int(audit["n_prompts"])
        nominal_total = n_prompts * (m_ref + 1)
        completed_samples = (
            counts["completed_reference_samples"] + counts["completed_target_samples"]
        )
        completed_prompts = min(
            n_prompts,
            counts["completed_target_samples"],
            counts["completed_reference_samples"] // max(1, m_ref),
        )
        events = self.repo.list_audit_events(audit_id)
        latest_event = events[-1] if events else None

        return {
            "audit_id": audit_id,
            "name": audit["name"],
            "is_demo": audit["is_demo"],
            "score_mode": audit["score_mode"],
            "execution_state": audit["execution_state"],
            "verdict": audit["verdict"],
            "verdict_reason": audit["verdict_reason"],
            "completed_prompts": completed_prompts,
            "total_prompts": n_prompts,
            "completed_samples": completed_samples,
            "nominal_total_samples": nominal_total,
            "progress_fraction": round(completed_samples / max(1, nominal_total), 4),
            "attempts": {
                **counts,
                "max_reference_attempts": audit["max_reference_attempts"],
                "max_target_attempts": audit["max_target_attempts"],
                "max_total_attempts": audit["max_total_attempts"],
            },
            "local_scoring_compute_ms": round(float(audit["local_scoring_compute_ms"]), 3),
            "resume_count": audit["resume_count"],
            "uncertain_prior_attempts": audit["uncertain_prior_attempts"],
            "latest_event": latest_event,
            "updated_at": audit["updated_at"],
        }

    def run_demo_calibration(self, req: DemoCalibrationCreateRequest) -> Dict[str, Any]:
        """Run actual seeded synthetic power simulations against substituted & mixed-routing alternatives."""
        probe_rev = (
            self.repo.get_probe_revision(req.probe_revision_id)
            if req.probe_revision_id
            else self.repo.get_default_probe_revision()
        )
        if not probe_rev:
            raise ValueError(f"Probe revision '{req.probe_revision_id}' not found.")

        selected_prompts = probe_rev["prompts"][: req.n_prompts]
        ref_adapter = SyntheticDemoAdapter(
            {
                "model_identifier": "demo-llama-3-8b-instruct-fp16",
                "synthetic_scenario": SyntheticScenario.GENUINE.value,
            }
        )

        alt_configs = [
            (
                "Substituted-INT4-Distilled-Sim",
                "Synthetic substituted model with systematic token rank and length shift.",
                SyntheticScenario.SUBSTITUTED.value,
                1.0,
            ),
            (
                "Mixed-Routing-65pct-Substitute-Sim",
                "Synthetic mixed-routing pool directing 65% of queries to the substitute model.",
                SyntheticScenario.MIXED_ROUTING.value,
                0.65,
            ),
        ]

        rng_master = np.random.default_rng(int(req.calibration_seed))
        alt_results: List[CalibrationAlternativeResult] = []

        for alt_name, alt_desc, scenario_val, bad_frac in alt_configs:
            tgt_adapter = SyntheticDemoAdapter(
                {
                    "model_identifier": alt_name,
                    "synthetic_scenario": scenario_val,
                    "mixed_routing_bad_fraction": bad_frac,
                }
            )
            rejected_count = 0
            for trial_idx in range(req.n_trials):
                trial_seed = int(rng_master.integers(10_000, 2_000_000_000))
                t_rng = np.random.default_rng(trial_seed)
                z_list: List[float] = []
                for p_obj in selected_prompts:
                    pid = str(p_obj["prompt_id"])
                    ptext = str(p_obj["prompt_text"])
                    ref_scores: List[float] = []
                    for _ in range(req.m_reference):
                        g_ref = ref_adapter.generate(
                            pid, ptext, 0.7, 0.95, req.max_output_tokens, t_rng
                        )
                        if req.score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY:
                            ref_scores.append(compute_unicode_length_score(g_ref.response_text))
                        else:
                            ref_scores.append(float(g_ref.precomputed_logit_rank or 0.0))

                    g_tgt = tgt_adapter.generate(
                        pid, ptext, 0.7, 0.95, req.max_output_tokens, t_rng
                    )
                    if req.score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY:
                        t_score = compute_unicode_length_score(g_tgt.response_text)
                    else:
                        t_score = float(g_tgt.precomputed_logit_rank or 0.0)

                    rz = compute_prompt_rank_z(
                        ref_scores, t_score, req.score_mode.value, t_rng
                    )
                    z_list.append(rz["z_uniform"])

                mc_res = run_cvm_monte_carlo_test(
                    z_list,
                    mc_null_seed=int(trial_seed ^ 0x5A5A5A),
                    n_simulations=9999,
                )
                if mc_res["p_value"] <= req.alpha:
                    rejected_count += 1

            emp_power = rejected_count / float(req.n_trials)
            w_low = wilson_lower_bound_95(rejected_count, req.n_trials)
            cp_low, _ = clopper_pearson_interval_95(rejected_count, req.n_trials)
            alt_results.append(
                CalibrationAlternativeResult(
                    alternative_name=alt_name,
                    alternative_description=alt_desc,
                    n_trials=req.n_trials,
                    n_rejected=rejected_count,
                    empirical_power=round(emp_power, 6),
                    wilson_lower_95=round(w_low, 6),
                    clopper_pearson_lower_95=round(cp_low, 6),
                )
            )

        demo_ref_prof = self.repo.get_profile(req.reference_profile_id) or {
            "id": "prof_demo_reference",
            "adapter_type": "synthetic_demo",
            "model_identifier": "demo-llama-3-8b-instruct-fp16",
            "precision": "float32",
            "device": "cpu",
        }
        demo_sc_prof = (
            self.repo.get_profile(req.scoring_reference_profile_id)
            if req.scoring_reference_profile_id
            else demo_ref_prof
        ) or demo_ref_prof
        demo_identity = build_reference_scorer_identity(demo_ref_prof, demo_sc_prof)

        import_req = CalibrationImportRequest(
            name=req.name,
            description="Generated by ModelTrace DEMO empirical power calibration runner.",
            score_mode=req.score_mode,
            probe_revision_id=probe_rev["id"],
            probe_revision_hash=probe_rev["content_hash"],
            n_prompts=req.n_prompts,
            m_reference=req.m_reference,
            max_output_tokens=req.max_output_tokens,
            alpha=req.alpha,
            temperature=req.temperature,
            top_p=req.top_p,
            reference_model_descriptor=str(demo_ref_prof.get("model_identifier") or "demo-llama-3-8b-instruct-fp16"),
            reference_profile_id=req.reference_profile_id,
            scoring_reference_profile_id=req.scoring_reference_profile_id,
            reference_identity=demo_identity,
            stat_engine_method=STAT_ENGINE_METHOD,
            stat_engine_version=STAT_ENGINE_VERSION,
            is_demo=True,
            provenance_notes=(
                f"Executed {req.n_trials} independent synthetic trials per named alternative "
                f"with B=9999 Monte Carlo uniform null simulations at alpha={req.alpha}."
            ),
            alternatives=alt_results,
            calibration_seed=req.calibration_seed,
        )
        return self.repo.import_calibration(import_req)
