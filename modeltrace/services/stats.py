"""Statistical verification engine for ModelTrace.

Implements:
- Deterministic SeedSequence stream derivation (reference, target, tie-break, MC null, bootstrap)
- Primary teacher-forcing mean log vocabulary rank scoring (LOCAL_LOGIT_RANK)
- Response-only Unicode length fallback scoring (LENGTH_DISTRIBUTION_ONLY)
- Finite-sample randomized tie-broken rank transformation z = (a + J + U) / (m + 1)
- Cramér-von Mises test statistic with B >= 9999 Monte Carlo uniform null simulations
- Holm step-down multiplicity correction for optional secondary inferential tests
- Prompt-level and task-family-level bootstrap confidence intervals for score shift
- Two-sided 95% Wilson score binomial confidence intervals AND exact Clopper-Pearson binomial intervals
- Strict calibration recomputation, full reference/scorer identity binding, exact budget matching, and >=80% 95% Wilson lower-confidence bound enforcement
"""

import math
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from modeltrace.schemas import AuditVerdict, ExecutionState, ScoreMode

STAT_ENGINE_METHOD = "cramer_von_mises_randomized_rank_mc"
STAT_ENGINE_VERSION = "1.0.0"
SCORE_QUANTIZATION_DECIMALS = 9


def _is_valid_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def derive_audit_seeds(master_seed: int) -> Dict[str, int]:
    """Derive 5 independent RNG seeds from master_seed using numpy.random.SeedSequence."""
    ss = np.random.SeedSequence(int(master_seed) & 0xFFFFFFFF)
    children = ss.spawn(5)
    names = [
        "reference_seed",
        "target_seed",
        "tie_break_seed",
        "mc_null_seed",
        "bootstrap_seed",
    ]
    out: Dict[str, int] = {"master_seed": int(master_seed)}
    for name, child in zip(names, children):
        out[name] = int(child.generate_state(1, dtype=np.uint32)[0])
    return out


def compute_unicode_length_score(text: str) -> float:
    """Compute frozen Unicode NFC codepoint length for LENGTH_DISTRIBUTION_ONLY mode."""
    normalized = unicodedata.normalize("NFC", text or "")
    return float(len(normalized))


def quantize_score(value: float, score_mode: str) -> float:
    """Apply frozen quantization grid so floating-point tie detection is transitive and exact."""
    if not _is_finite_number(value):
        raise ValueError(f"Score value must be finite; got {value!r}.")
    if score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value:
        return float(int(round(value)))
    return round(float(value), SCORE_QUANTIZATION_DECIMALS)


def compute_prompt_rank_z(
    reference_scores: Sequence[float],
    target_score: float,
    score_mode: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Compute finite-sample tie-broken uniformized rank z = (a + J + U) / (m + 1)."""
    m = len(reference_scores)
    if m < 1:
        raise ValueError("At least m >= 1 reference score is required per prompt.")

    q_refs = [quantize_score(r, score_mode) for r in reference_scores]
    q_tgt = quantize_score(target_score, score_mode)

    a_below = sum(1 for r in q_refs if r < q_tgt)
    b_ties = sum(1 for r in q_refs if r == q_tgt)

    j_draw = int(rng.integers(0, b_ties + 1))
    u_draw = float(rng.random())
    z_uniform = float((a_below + j_draw + u_draw) / (m + 1))
    mean_ref = float(np.mean(q_refs))
    score_shift = float(q_tgt - mean_ref)

    return {
        "m_reference": m,
        "quantized_reference_scores": q_refs,
        "quantized_target_score": q_tgt,
        "a_below": int(a_below),
        "b_ties": int(b_ties),
        "j_draw": j_draw,
        "u_draw": u_draw,
        "z_uniform": z_uniform,
        "score_shift": score_shift,
    }


def compute_cvm_statistic(z_values: Sequence[float]) -> float:
    """Compute the Cramér-von Mises statistic W^2 against Uniform(0, 1)."""
    z_arr = np.asarray(z_values, dtype=np.float64)
    n = z_arr.size
    if n < 1:
        raise ValueError("Cannot compute Cramér-von Mises statistic on empty z array.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("Cannot compute Cramér-von Mises statistic on non-finite z values.")
    z_sorted = np.sort(np.clip(z_arr, 0.0, 1.0))
    k = np.arange(1, n + 1, dtype=np.float64)
    expected = (2.0 * k - 1.0) / (2.0 * n)
    w2 = (1.0 / (12.0 * n)) + float(np.sum((z_sorted - expected) ** 2))
    return float(w2)


def run_cvm_monte_carlo_test(
    z_values: Sequence[float],
    mc_null_seed: int,
    n_simulations: int = 9999,
) -> Dict[str, Any]:
    """Compute Cramér-von Mises statistic and finite-sample Monte Carlo p-value."""
    if n_simulations < 9999:
        raise ValueError(f"n_simulations (B) must be >= 9999; got {n_simulations}.")
    z_arr = np.asarray(z_values, dtype=np.float64)
    n = int(z_arr.size)
    if n < 1:
        raise ValueError("At least 1 prompt z-value is required.")

    obs_stat = compute_cvm_statistic(z_arr)
    rng = np.random.default_rng(int(mc_null_seed))

    null_samples = rng.random((n_simulations, n), dtype=np.float64)
    null_samples.sort(axis=1)
    k = np.arange(1, n + 1, dtype=np.float64)
    expected = (2.0 * k - 1.0) / (2.0 * n)
    null_stats = (1.0 / (12.0 * n)) + np.sum((null_samples - expected) ** 2, axis=1)

    exceedances = int(np.sum(null_stats >= (obs_stat - 1e-12)))
    p_value = float((1 + exceedances) / (n_simulations + 1))

    return {
        "method": STAT_ENGINE_METHOD,
        "version": STAT_ENGINE_VERSION,
        "n_prompts": n,
        "mc_simulations": int(n_simulations),
        "mc_null_seed": int(mc_null_seed),
        "cvm_statistic": obs_stat,
        "exceedances": exceedances,
        "p_value": p_value,
        "null_quantiles": {
            "q50": float(np.quantile(null_stats, 0.50)),
            "q90": float(np.quantile(null_stats, 0.90)),
            "q95": float(np.quantile(null_stats, 0.95)),
            "q99": float(np.quantile(null_stats, 0.99)),
        },
    }


def holm_step_down(
    tests: List[Dict[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """Apply Holm-Bonferroni step-down multiplicity correction to a list of inferential tests."""
    k_total = len(tests)
    if k_total == 0:
        return []

    indexed = list(enumerate(tests))
    indexed.sort(key=lambda pair: float(pair[1]["raw_p_value"]))

    adjusted_by_orig_idx: Dict[int, Dict[str, Any]] = {}
    running_max = 0.0
    still_rejecting = True

    for rank_idx, (orig_idx, item) in enumerate(indexed):
        raw_p = float(item["raw_p_value"])
        multiplier = k_total - rank_idx
        adj_p = min(1.0, max(running_max, raw_p * multiplier))
        running_max = adj_p
        if still_rejecting and adj_p <= alpha:
            rejected = True
        else:
            still_rejecting = False
            rejected = False

        updated = dict(item)
        updated["holm_adjusted_p_value"] = float(adj_p)
        updated["rejected_at_alpha"] = rejected
        updated["correction_method"] = "holm_bonferroni_step_down"
        adjusted_by_orig_idx[orig_idx] = updated

    return [adjusted_by_orig_idx[i] for i in range(k_total)]


def compute_bootstrap_uncertainty(
    prompt_records: Sequence[Dict[str, Any]],
    bootstrap_seed: int,
    n_replicates: int = 1000,
) -> Dict[str, Any]:
    """Estimate score-shift and rank-shift 95% confidence intervals via prompt & family bootstrap."""
    n = len(prompt_records)
    if n == 0:
        return {
            "n_replicates": 0,
            "mean_score_shift": 0.0,
            "prompt_bootstrap_ci_95": [0.0, 0.0],
            "family_bootstrap_ci_95": [0.0, 0.0],
            "mean_z_shift_from_half": 0.0,
            "z_shift_ci_95": [0.0, 0.0],
            "by_family_descriptive": {},
        }

    shifts = np.array([float(r["score_shift"]) for r in prompt_records], dtype=np.float64)
    z_vals = np.array([float(r["z_uniform"]) - 0.5 for r in prompt_records], dtype=np.float64)
    mean_shift = float(np.mean(shifts))
    mean_z_shift = float(np.mean(z_vals))

    rng = np.random.default_rng(int(bootstrap_seed))
    if n == 1:
        ci_prompt = [mean_shift, mean_shift]
        ci_z = [mean_z_shift, mean_z_shift]
    else:
        idx = rng.integers(0, n, size=(n_replicates, n))
        boot_means = np.mean(shifts[idx], axis=1)
        boot_z_means = np.mean(z_vals[idx], axis=1)
        ci_prompt = [
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
        ]
        ci_z = [
            float(np.quantile(boot_z_means, 0.025)),
            float(np.quantile(boot_z_means, 0.975)),
        ]

    family_map: Dict[str, List[float]] = {}
    for r in prompt_records:
        fam = str(r.get("family", "general"))
        family_map.setdefault(fam, []).append(float(r["score_shift"]))

    families = sorted(family_map.keys())
    by_family_desc: Dict[str, Any] = {}
    for fam in families:
        arr = np.array(family_map[fam], dtype=np.float64)
        by_family_desc[fam] = {
            "n_prompts": int(arr.size),
            "mean_score_shift": float(np.mean(arr)),
            "median_score_shift": float(np.median(arr)),
            "scope": "DESCRIPTIVE_ONLY",
        }

    if len(families) <= 1:
        ci_family = list(ci_prompt)
    else:
        fam_arrays = [np.array(family_map[f], dtype=np.float64) for f in families]
        n_fams = len(fam_arrays)
        fam_boot_means = np.empty(n_replicates, dtype=np.float64)
        for rep in range(n_replicates):
            chosen_fams = rng.integers(0, n_fams, size=n_fams)
            pooled: List[float] = []
            for f_idx in chosen_fams:
                f_arr = fam_arrays[int(f_idx)]
                sub_idx = rng.integers(0, f_arr.size, size=f_arr.size)
                pooled.extend(f_arr[sub_idx].tolist())
            fam_boot_means[rep] = float(np.mean(pooled))
        ci_family = [
            float(np.quantile(fam_boot_means, 0.025)),
            float(np.quantile(fam_boot_means, 0.975)),
        ]

    return {
        "n_replicates": int(n_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "mean_score_shift": mean_shift,
        "median_score_shift": float(np.median(shifts)),
        "prompt_bootstrap_ci_95": ci_prompt,
        "family_bootstrap_ci_95": ci_family,
        "mean_z_shift_from_half": mean_z_shift,
        "z_shift_ci_95": ci_z,
        "by_family_descriptive": by_family_desc,
    }


def wilson_interval_95(n_success: int, n_trials: int) -> Tuple[float, float]:
    """Compute two-sided 95% Wilson score confidence interval (lower, upper) for a binomial proportion.

    Note: Wilson score intervals invert the normal-theory score test (Wilson, 1927); they are distinct from
    exact Clopper-Pearson intervals (which invert the exact binomial CDF via Beta quantiles).
    """
    if n_trials <= 0 or n_success < 0 or n_success > n_trials:
        raise ValueError(f"Invalid binomial counts n_success={n_success}, n_trials={n_trials}.")
    z = float(scipy_stats.norm.ppf(0.975))
    phat = float(n_success) / float(n_trials)
    denom = 1.0 + (z * z) / n_trials
    center = phat + (z * z) / (2.0 * n_trials)
    half_width = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n_trials)) / n_trials)
    lower = max(0.0, float((center - half_width) / denom))
    upper = min(1.0, float((center + half_width) / denom))
    return (lower, upper)


def wilson_lower_bound_95(n_success: int, n_trials: int) -> float:
    """Compute two-sided 95% Wilson score confidence interval lower bound."""
    return wilson_interval_95(n_success, n_trials)[0]


def clopper_pearson_interval_95(n_success: int, n_trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute exact two-sided Clopper-Pearson (1 - alpha) binomial confidence interval via Beta quantiles."""
    if n_trials <= 0 or n_success < 0 or n_success > n_trials:
        raise ValueError(f"Invalid binomial counts n_success={n_success}, n_trials={n_trials}.")
    if n_success == 0:
        lower = 0.0
    else:
        lower = float(scipy_stats.beta.ppf(alpha / 2.0, n_success, n_trials - n_success + 1))
    if n_success == n_trials:
        upper = 1.0
    else:
        upper = float(scipy_stats.beta.ppf(1.0 - alpha / 2.0, n_success + 1, n_trials - n_success))
    return (max(0.0, lower), min(1.0, upper))


def build_reference_scorer_identity(
    reference_profile: Optional[Dict[str, Any]],
    scoring_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract canonical immutable identity of reference generator and scorer profiles."""
    if not reference_profile:
        return {}
    scorer = scoring_profile or reference_profile
    return {
        "reference_adapter_type": str(reference_profile.get("adapter_type") or ""),
        "reference_model_identifier": str(reference_profile.get("model_identifier") or ""),
        "reference_base_url": str(reference_profile.get("base_url") or ""),
        "reference_checkpoint_path": str(reference_profile.get("checkpoint_path") or ""),
        "reference_checkpoint_revision": str(reference_profile.get("checkpoint_revision") or "main"),
        "reference_tokenizer_path": str(
            reference_profile.get("tokenizer_path") or reference_profile.get("checkpoint_path") or ""
        ),
        "reference_chat_template": str(reference_profile.get("chat_template") or ""),
        "reference_precision": str(reference_profile.get("precision") or "float32"),
        "reference_device": str(reference_profile.get("device") or "cpu"),
        "scorer_adapter_type": str(scorer.get("adapter_type") or ""),
        "scorer_model_identifier": str(scorer.get("model_identifier") or ""),
        "scorer_checkpoint_path": str(scorer.get("checkpoint_path") or ""),
        "scorer_checkpoint_revision": str(scorer.get("checkpoint_revision") or "main"),
        "scorer_tokenizer_path": str(
            scorer.get("tokenizer_path") or scorer.get("checkpoint_path") or ""
        ),
        "scorer_chat_template": str(scorer.get("chat_template") or ""),
        "scorer_precision": str(scorer.get("precision") or "float32"),
        "scorer_device": str(scorer.get("device") or "cpu"),
    }


def validate_and_recompute_calibration_alternatives(
    alternatives: Any,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Validate trial/rejection counts, recompute empirical power and Wilson/Clopper-Pearson bounds, and reject forged values."""
    if not isinstance(alternatives, list) or len(alternatives) == 0:
        raise ValueError("Calibration artifact must contain a non-empty list of 'alternatives'.")

    validated_alts: List[Dict[str, Any]] = []
    for idx, raw_alt in enumerate(alternatives):
        if not isinstance(raw_alt, dict):
            raise ValueError(f"Alternative at index {idx} must be a dictionary object.")
        alt_name = str(raw_alt.get("alternative_name") or "").strip()
        if not alt_name:
            raise ValueError(f"Alternative at index {idx} is missing a valid 'alternative_name'.")

        raw_trials = raw_alt.get("n_trials")
        raw_rej = raw_alt.get("n_rejected")
        if raw_rej is None and "rejections" in raw_alt:
            raw_rej = raw_alt.get("rejections")

        if not _is_valid_int(raw_trials) or int(raw_trials) < 1:
            raise ValueError(
                f"Alternative '{alt_name}' has invalid n_trials={raw_trials!r}; must be an integer >= 1."
            )
        if not _is_valid_int(raw_rej) or int(raw_rej) < 0 or int(raw_rej) > int(raw_trials):
            raise ValueError(
                f"Alternative '{alt_name}' has invalid n_rejected={raw_rej!r} for n_trials={raw_trials} "
                "(integer count 0 <= n_rejected <= n_trials required)."
            )

        n_trials = int(raw_trials)
        n_rejected = int(raw_rej)
        recomputed_power = n_rejected / float(n_trials)
        w_low, w_high = wilson_interval_95(n_rejected, n_trials)
        cp_low, cp_high = clopper_pearson_interval_95(n_rejected, n_trials)

        supplied_power = raw_alt.get("empirical_power")
        if supplied_power is not None:
            if not _is_finite_number(supplied_power):
                raise ValueError(f"Alternative '{alt_name}' has non-finite empirical_power={supplied_power!r}.")
            if abs(float(supplied_power) - recomputed_power) > tolerance:
                raise ValueError(
                    f"Alternative '{alt_name}' empirical_power ({supplied_power}) is inconsistent with "
                    f"n_rejected/n_trials ({n_rejected}/{n_trials} = {recomputed_power:.6f})."
                )

        supplied_wilson = raw_alt.get("wilson_lower_95")
        if supplied_wilson is not None:
            if not _is_finite_number(supplied_wilson):
                raise ValueError(f"Alternative '{alt_name}' has non-finite wilson_lower_95={supplied_wilson!r}.")
            if abs(float(supplied_wilson) - w_low) > tolerance:
                raise ValueError(
                    f"Alternative '{alt_name}' supplied wilson_lower_95 ({supplied_wilson}) is forged or inconsistent "
                    f"with recomputed Wilson 95% lower bound ({w_low:.7f}) for {n_rejected}/{n_trials}."
                )

        supplied_cp = raw_alt.get("clopper_pearson_lower_95")
        if supplied_cp is not None:
            if not _is_finite_number(supplied_cp):
                raise ValueError(
                    f"Alternative '{alt_name}' has non-finite clopper_pearson_lower_95={supplied_cp!r}."
                )
            if abs(float(supplied_cp) - cp_low) > tolerance:
                raise ValueError(
                    f"Alternative '{alt_name}' supplied clopper_pearson_lower_95 ({supplied_cp}) is inconsistent "
                    f"with recomputed Clopper-Pearson 95% lower bound ({cp_low:.7f}) for {n_rejected}/{n_trials}."
                )

        validated_alts.append(
            {
                "alternative_name": alt_name,
                "alternative_description": str(raw_alt.get("alternative_description") or ""),
                "n_trials": n_trials,
                "n_rejected": n_rejected,
                "empirical_power": round(recomputed_power, 6),
                "wilson_lower_95": round(w_low, 6),
                "wilson_ci_95": [round(w_low, 6), round(w_high, 6)],
                "clopper_pearson_lower_95": round(cp_low, 6),
                "clopper_pearson_ci_95": [round(cp_low, 6), round(cp_high, 6)],
            }
        )

    min_power = min(a["empirical_power"] for a in validated_alts)
    min_trials = min(a["n_trials"] for a in validated_alts)
    min_wilson_lower = min(a["wilson_lower_95"] for a in validated_alts)
    min_cp_lower = min(a["clopper_pearson_lower_95"] for a in validated_alts)

    return {
        "alternatives": validated_alts,
        "min_empirical_power": min_power,
        "min_trials": min_trials,
        "min_wilson_lower_95": min_wilson_lower,
        "min_clopper_pearson_lower_95": min_cp_lower,
    }


def check_calibration_compatibility(
    calibration: Optional[Dict[str, Any]],
    score_mode: str,
    probe_revision_hash: str,
    n_prompts: int,
    m_reference: int,
    alpha: float,
    audit_is_demo: bool,
    probe_revision_id: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    reference_model_descriptor: Optional[str] = None,
    reference_identity: Optional[Dict[str, Any]] = None,
    stat_engine_method: str = STAT_ENGINE_METHOD,
    stat_engine_version: str = STAT_ENGINE_VERSION,
    min_required_power: float = 0.80,
    min_required_trials: int = 20,
    min_required_wilson_lower_95: float = 0.80,
) -> Dict[str, Any]:
    """Verify whether an attached calibration artifact qualifies non-rejection for CONSISTENT_WITHIN_TESTED_LIMITS.

    Enforces:
    - Recomputed empirical_power and 95% Wilson / Clopper-Pearson bounds from integer (n_rejected, n_trials) counts
    - Rejection of empty, malformed, non-finite, or forged calibration fields
    - Exact match on score_mode, probe_revision_hash (and probe_revision_id when provided), stat_engine_method, and stat_engine_version
    - Exact match on validated budget (n_prompts == cal.n_prompts and m_reference == cal.m_reference; no automatic transfer to unvalidated higher/lower budgets)
    - Exact match on alpha, temperature, top_p, and max_output_tokens
    - Exact match on is_demo provenance and full reference/scorer identity (model_identifier, checkpoint_path, revision, tokenizer_path, chat_template, precision, device)
    - Strict lower-confidence bound requirement matching the declared power target: min_trials >= 20, recomputed empirical_power >= 0.80, AND recomputed two-sided 95% Wilson lower bound >= 0.80 for every covered alternative.
    """
    if not calibration or not isinstance(calibration, dict):
        return {
            "compatible": False,
            "reason": "No power calibration artifact attached. Non-rejection without established sensitivity defaults to INCONCLUSIVE.",
        }

    issues: List[str] = []

    for num_key in ("n_prompts", "m_reference", "alpha"):
        if not _is_finite_number(calibration.get(num_key)):
            issues.append(f"calibration field '{num_key}' is missing or non-finite ({calibration.get(num_key)!r})")
    for opt_num_key in ("temperature", "top_p", "max_output_tokens", "min_empirical_power", "min_trials"):
        if calibration.get(opt_num_key) is not None and not _is_finite_number(calibration.get(opt_num_key)):
            issues.append(f"calibration field '{opt_num_key}' is non-finite ({calibration.get(opt_num_key)!r})")

    recomputed_summary: Optional[Dict[str, Any]] = None
    try:
        recomputed_summary = validate_and_recompute_calibration_alternatives(
            calibration.get("alternatives")
        )
        cal_min_p = calibration.get("min_empirical_power")
        if cal_min_p is not None and _is_finite_number(cal_min_p):
            if abs(float(cal_min_p) - recomputed_summary["min_empirical_power"]) > 1e-3:
                issues.append(
                    f"calibration min_empirical_power ({cal_min_p}) does not match recomputed minimum power ({recomputed_summary['min_empirical_power']:.4f})"
                )
        cal_min_t = calibration.get("min_trials")
        if cal_min_t is not None and _is_valid_int(cal_min_t):
            if int(cal_min_t) != recomputed_summary["min_trials"]:
                issues.append(
                    f"calibration min_trials ({cal_min_t}) does not match recomputed minimum trials ({recomputed_summary['min_trials']})"
                )
    except ValueError as exc:
        issues.append(f"invalid calibration alternatives: {exc}")

    if calibration.get("score_mode") != score_mode:
        issues.append(
            f"score_mode mismatch (calibration='{calibration.get('score_mode')}', audit='{score_mode}')"
        )
    if calibration.get("probe_revision_hash") != probe_revision_hash:
        issues.append("probe_revision_hash mismatch between calibration and audit manifest")
    if (
        probe_revision_id is not None
        and calibration.get("probe_revision_id")
        and calibration.get("probe_revision_id") != probe_revision_id
    ):
        issues.append(
            f"probe_revision_id mismatch (calibration='{calibration.get('probe_revision_id')}', audit='{probe_revision_id}')"
        )

    cal_method = calibration.get("stat_engine_method") or STAT_ENGINE_METHOD
    cal_version = calibration.get("stat_engine_version") or STAT_ENGINE_VERSION
    if cal_method != stat_engine_method:
        issues.append(
            f"stat_engine_method mismatch (calibration='{cal_method}', audit='{stat_engine_method}')"
        )
    if cal_version != stat_engine_version:
        issues.append(
            f"stat_engine_version mismatch (calibration='{cal_version}', audit='{stat_engine_version}')"
        )

    if _is_finite_number(calibration.get("n_prompts")) and _is_finite_number(calibration.get("m_reference")):
        cal_n = int(calibration["n_prompts"])
        cal_m = int(calibration["m_reference"])
        if cal_n != int(n_prompts):
            issues.append(
                f"budget n_prompts mismatch: calibration validated for n_prompts={cal_n}, audit requested n_prompts={n_prompts} (no automatic budget transfer without validated coverage)"
            )
        if cal_m != int(m_reference):
            issues.append(
                f"budget m_reference mismatch: calibration validated for m_reference={cal_m}, audit requested m_reference={m_reference} (no automatic budget transfer without validated coverage)"
            )

    if _is_finite_number(calibration.get("alpha")):
        if abs(float(calibration["alpha"]) - float(alpha)) > 1e-6:
            issues.append(
                f"alpha mismatch (calibration={calibration.get('alpha')}, audit={alpha})"
            )

    if temperature is not None and calibration.get("temperature") is not None and _is_finite_number(calibration.get("temperature")):
        if abs(float(calibration["temperature"]) - float(temperature)) > 1e-6:
            issues.append(
                f"temperature mismatch (calibration={calibration.get('temperature')}, audit={temperature})"
            )

    if top_p is not None and calibration.get("top_p") is not None and _is_finite_number(calibration.get("top_p")):
        if abs(float(calibration["top_p"]) - float(top_p)) > 1e-6:
            issues.append(
                f"top_p mismatch (calibration={calibration.get('top_p')}, audit={top_p})"
            )

    if (
        max_output_tokens is not None
        and calibration.get("max_output_tokens") is not None
        and _is_finite_number(calibration.get("max_output_tokens"))
    ):
        if int(calibration["max_output_tokens"]) != int(max_output_tokens):
            issues.append(
                f"max_output_tokens mismatch (calibration={calibration.get('max_output_tokens')}, audit={max_output_tokens})"
            )

    if bool(calibration.get("is_demo")) != bool(audit_is_demo):
        issues.append(
            f"is_demo provenance mismatch (calibration is_demo={bool(calibration.get('is_demo'))}, audit is_demo={bool(audit_is_demo)})"
        )

    if (
        reference_model_descriptor is not None
        and calibration.get("reference_model_descriptor")
        and str(calibration.get("reference_model_descriptor")) != str(reference_model_descriptor)
    ):
        issues.append(
            f"reference_model_descriptor mismatch (calibration='{calibration.get('reference_model_descriptor')}', audit='{reference_model_descriptor}')"
        )

    cal_identity = calibration.get("reference_identity")
    if isinstance(cal_identity, dict) and cal_identity and isinstance(reference_identity, dict) and reference_identity:
        for id_key in (
            "reference_adapter_type",
            "reference_model_identifier",
            "reference_base_url",
            "reference_checkpoint_path",
            "reference_checkpoint_revision",
            "reference_tokenizer_path",
            "reference_chat_template",
            "reference_precision",
            "reference_device",
            "scorer_adapter_type",
            "scorer_model_identifier",
            "scorer_checkpoint_path",
            "scorer_checkpoint_revision",
            "scorer_tokenizer_path",
            "scorer_chat_template",
            "scorer_precision",
            "scorer_device",
        ):
            if id_key in cal_identity and id_key in reference_identity:
                if str(cal_identity[id_key] or "") != str(reference_identity[id_key] or ""):
                    issues.append(
                        f"reference/scorer identity mismatch on '{id_key}' (calibration='{cal_identity[id_key]}', audit='{reference_identity[id_key]}')"
                    )

    if recomputed_summary is not None:
        if recomputed_summary["min_trials"] < min_required_trials:
            issues.append(
                f"calibration trial count ({recomputed_summary['min_trials']}) is below minimum required ({min_required_trials})"
            )
        if recomputed_summary["min_empirical_power"] < min_required_power:
            issues.append(
                f"calibration recomputed minimum empirical sensitivity ({recomputed_summary['min_empirical_power']:.4f}) is below threshold ({min_required_power:.2f})"
            )
        if recomputed_summary["min_wilson_lower_95"] < min_required_wilson_lower_95:
            issues.append(
                f"calibration recomputed 95% Wilson lower confidence bound ({recomputed_summary['min_wilson_lower_95']:.4f}) is below required lower-confidence bound ({min_required_wilson_lower_95:.2f}) for covered alternatives"
            )

    if issues:
        return {
            "compatible": False,
            "calibration_id": calibration.get("id"),
            "is_demo_calibration": bool(calibration.get("is_demo")),
            "issues": issues,
            "reason": "Attached calibration is incompatible: " + "; ".join(issues),
        }

    assert recomputed_summary is not None
    return {
        "compatible": True,
        "calibration_id": calibration.get("id"),
        "calibration_name": calibration.get("name"),
        "is_demo_calibration": bool(calibration.get("is_demo")),
        "min_empirical_power": recomputed_summary["min_empirical_power"],
        "min_trials": recomputed_summary["min_trials"],
        "min_wilson_lower_95": recomputed_summary["min_wilson_lower_95"],
        "min_clopper_pearson_lower_95": recomputed_summary["min_clopper_pearson_lower_95"],
        "alternatives": recomputed_summary["alternatives"],
        "reason": (
            f"Verified compatible calibration '{calibration.get('name')}' "
            f"(recomputed min empirical power {recomputed_summary['min_empirical_power']:.1%}, "
            f"two-sided 95% Wilson lower bound {recomputed_summary['min_wilson_lower_95']:.4f} >= {min_required_wilson_lower_95:.2f}, "
            f"exact two-sided 95% Clopper-Pearson lower bound {recomputed_summary['min_clopper_pearson_lower_95']:.4f} "
            f"across >= {recomputed_summary['min_trials']} trials)."
        ),
    }


def evaluate_verdict(
    execution_state: str,
    completed_prompts: int,
    expected_n_prompts: int,
    all_prompts_have_full_samples: bool,
    p_value: Optional[float],
    alpha: float,
    score_mode: str,
    calibration_check: Dict[str, Any],
    is_demo: bool,
) -> Tuple[str, str]:
    """Determine (verdict, verdict_reason) strictly decoupled from execution_state."""
    demo_prefix = "[DEMO SYNTHETIC] " if is_demo else ""
    mode_scope = (
        " [LENGTH DISTRIBUTION ONLY — cannot verify token-level identity]"
        if score_mode == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value
        else ""
    )

    if (
        execution_state != ExecutionState.COMPLETED.value
        or completed_prompts < expected_n_prompts
        or not all_prompts_have_full_samples
        or p_value is None
    ):
        return (
            AuditVerdict.INCONCLUSIVE.value,
            f"{demo_prefix}Execution state is {execution_state} with {completed_prompts}/{expected_n_prompts} complete prompts. "
            f"Incomplete or invalid samples cannot yield a valid inferential verdict.{mode_scope}",
        )

    if p_value <= alpha:
        return (
            AuditVerdict.INCONSISTENT.value,
            f"{demo_prefix}Tested behavior rejects the null hypothesis of exchangeability at alpha={alpha:.4f} "
            f"(Monte Carlo CvM p={p_value:.5f}). Note: behavioral divergence may reflect model substitution, "
            f"quantization, chat template differences, hidden system prompts, or decoding differences.{mode_scope}",
        )

    if calibration_check.get("compatible"):
        return (
            AuditVerdict.CONSISTENT_WITHIN_TESTED_LIMITS.value,
            f"{demo_prefix}Non-rejection at alpha={alpha:.4f} (p={p_value:.5f}) backed by compatible power calibration "
            f"({calibration_check.get('reason')}). Consistency holds only within tested prompts, sampling limits, "
            f"and calibrated alternatives—not proof of weights or provider intent.{mode_scope}",
        )

    return (
        AuditVerdict.INCONCLUSIVE.value,
        f"{demo_prefix}Null hypothesis not rejected at alpha={alpha:.4f} (p={p_value:.5f}), but verdict is INCONCLUSIVE "
        f"because sensitivity at this budget is uncalibrated ({calibration_check.get('reason')}).{mode_scope}",
    )
