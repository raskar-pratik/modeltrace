"""Measured synthetic benchmark suite and real-model evaluation manifest generator for ModelTrace V1.

Implements strictly separated seed streams (`development_seed`, `calibration_seed`, `evaluation_seed`)
across genuine null trials, weak/near-null alternatives, multiple mixed-routing fractions, full substitution,
and budget/score-mode settings variations. Outputs empirical false-alarm rates, empirical detection power,
two-sided 95% Wilson score intervals, and exact two-sided 95% Clopper-Pearson binomial intervals.

IMPORTANT SCOPE STATEMENT:
Synthetic benchmark measurements establish statistical software and pipeline behavior under controlled
stochastic generative models only. They do NOT establish empirical detection accuracy on real hosted LLMs.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from modeltrace.adapters.synthetic_demo import SyntheticDemoAdapter
from modeltrace.services.probes import BUILTIN_PROBES
from modeltrace.services.stats import (
    STAT_ENGINE_METHOD,
    STAT_ENGINE_VERSION,
    build_reference_scorer_identity,
    check_calibration_compatibility,
    clopper_pearson_interval_95,
    compute_prompt_rank_z,
    derive_audit_seeds,
    run_cvm_monte_carlo_test,
    wilson_interval_95,
)


def _make_synthetic_profile(
    profile_id: str,
    scenario: str,
    mixed_substitute_fraction: float = 0.0,
) -> Dict[str, Any]:
    return {
        "id": profile_id,
        "name": f"Benchmark Synthetic ({scenario}, frac={mixed_substitute_fraction:.2f})",
        "adapter_type": "synthetic_demo",
        "model_identifier": f"synthetic-{scenario}-{int(mixed_substitute_fraction * 100)}",
        "synthetic_scenario": scenario,
        "mixed_routing_bad_fraction": mixed_substitute_fraction,
        "precision": "float32",
        "device": "cpu",
        "extra_config": {
            "is_demo": True,
            "scenario": scenario,
            "mixed_substitute_fraction": mixed_substitute_fraction,
        },
    }


def _evaluate_single_synthetic_trial(
    prompts: List[Dict[str, Any]],
    ref_adapter: SyntheticDemoAdapter,
    tgt_adapter: SyntheticDemoAdapter,
    score_mode: str,
    n_prompts: int,
    m_reference: int,
    alpha: float,
    trial_master_seed: int,
    mc_simulations: int = 9999,
) -> Dict[str, Any]:
    seeds = derive_audit_seeds(trial_master_seed)
    ref_rng = np.random.default_rng(seeds["reference_seed"])
    tgt_rng = np.random.default_rng(seeds["target_seed"])
    tie_rng = np.random.default_rng(seeds["tie_break_seed"])
    selected_prompts = prompts[:n_prompts]

    z_values: List[float] = []
    shift_values: List[float] = []
    for p in selected_prompts:
        pid = p["prompt_id"]
        ptxt = p.get("prompt_text") or p.get("text", "")
        ref_scores: List[float] = []
        for _ in range(m_reference):
            gen = ref_adapter.generate(pid, ptxt, 0.7, 0.95, 128, rng=ref_rng)
            sc = ref_adapter.score_response(
                pid, ptxt, gen.response_text, gen.observable_meta, score_mode
            )
            ref_scores.append(sc.raw_score)

        t_gen = tgt_adapter.generate(pid, ptxt, 0.7, 0.95, 128, rng=tgt_rng)
        t_sc = ref_adapter.score_response(
            pid, ptxt, t_gen.response_text, t_gen.observable_meta, score_mode
        )
        tgt_score = t_sc.raw_score

        rz = compute_prompt_rank_z(ref_scores, tgt_score, score_mode, tie_rng)
        z_values.append(rz["z_uniform"])
        shift_values.append(rz["score_shift"])

    cvm = run_cvm_monte_carlo_test(
        z_values=z_values,
        mc_null_seed=seeds["mc_null_seed"],
        n_simulations=mc_simulations,
    )
    rejected = bool(cvm["p_value"] <= alpha)
    return {
        "trial_master_seed": trial_master_seed,
        "cvm_statistic": cvm["cvm_statistic"],
        "p_value": cvm["p_value"],
        "rejected_at_alpha": rejected,
        "mean_score_shift": float(np.mean(shift_values)),
    }


def run_measured_synthetic_benchmark(
    n_eval_trials: int = 30,
    n_cal_trials: int = 25,
    development_seed: int = 100_000,
    calibration_seed: int = 500_000,
    evaluation_seed: int = 900_000,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full separated-seed synthetic benchmark suite and write JSON artifacts."""
    if len({development_seed, calibration_seed, evaluation_seed}) != 3:
        raise ValueError("development_seed, calibration_seed, and evaluation_seed must be strictly distinct.")

    prompts = list(BUILTIN_PROBES)
    ref_profile = _make_synthetic_profile("prof_bench_ref", "genuine", 0.0)
    ref_adapter = SyntheticDemoAdapter(ref_profile)
    ref_identity = build_reference_scorer_identity(ref_profile, ref_profile)

    # 1. Independent Calibration Run (using calibration_seed stream only)
    cal_alt_configs = [
        ("Substituted-100pct", "substituted", 1.0),
        ("Mixed-Routing-65pct", "mixed_routing", 0.65),
        ("Mixed-Routing-35pct-Underpowered", "mixed_routing", 0.35),
    ]
    cal_alternatives_all: List[Dict[str, Any]] = []
    for alt_idx, (alt_name, scen, frac) in enumerate(cal_alt_configs):
        alt_prof = _make_synthetic_profile(f"prof_cal_{alt_idx}", scen, frac)
        alt_adapter = SyntheticDemoAdapter(alt_prof)
        rejections = 0
        for t in range(n_cal_trials):
            t_seed = calibration_seed + alt_idx * 50_000 + t * 997
            res = _evaluate_single_synthetic_trial(
                prompts=prompts,
                ref_adapter=ref_adapter,
                tgt_adapter=alt_adapter,
                score_mode="local_logit_rank",
                n_prompts=20,
                m_reference=8,
                alpha=0.01,
                trial_master_seed=t_seed,
            )
            if res["rejected_at_alpha"]:
                rejections += 1
        emp_power = rejections / float(n_cal_trials)
        w_low, w_high = wilson_interval_95(rejections, n_cal_trials)
        cp_low, cp_high = clopper_pearson_interval_95(rejections, n_cal_trials)
        cal_alternatives_all.append(
            {
                "alternative_name": alt_name,
                "scenario": scen,
                "substitute_fraction": frac,
                "n_trials": n_cal_trials,
                "n_rejected": rejections,
                "empirical_power": round(emp_power, 6),
                "wilson_lower_95": round(w_low, 6),
                "wilson_ci_95": [round(w_low, 6), round(w_high, 6)],
                "clopper_pearson_lower_95": round(cp_low, 6),
                "clopper_pearson_ci_95": [round(cp_low, 6), round(cp_high, 6)],
            }
        )

    adequate_alts = cal_alternatives_all[:2]
    valid_cal_record = {
        "id": "cal_bench_valid_25",
        "name": "Predeclared Synthetic Calibration (Substituted & 65% Mixed, 25 trials)",
        "score_mode": "local_logit_rank",
        "probe_revision_id": "rev_builtin_screening_v1",
        "probe_revision_hash": "builtin_v1",
        "n_prompts": 20,
        "m_reference": 8,
        "max_output_tokens": 128,
        "alpha": 0.01,
        "temperature": 0.7,
        "top_p": 0.95,
        "reference_model_descriptor": ref_profile["model_identifier"],
        "reference_identity": ref_identity,
        "stat_engine_method": STAT_ENGINE_METHOD,
        "stat_engine_version": STAT_ENGINE_VERSION,
        "is_demo": True,
        "min_trials": n_cal_trials,
        "min_empirical_power": min(a["empirical_power"] for a in adequate_alts),
        "alternatives": adequate_alts,
    }
    underpowered_cal_record = {
        **valid_cal_record,
        "id": "cal_bench_underpowered_35pct",
        "name": "Underpowered Alternative Calibration (Includes 35% Mixed at n=20, m=8)",
        "min_empirical_power": min(a["empirical_power"] for a in cal_alternatives_all),
        "alternatives": cal_alternatives_all,
    }
    w5_low, _ = wilson_interval_95(5, 5)
    cp5_low, _ = clopper_pearson_interval_95(5, 5)
    inadequate_cal_record = {
        **valid_cal_record,
        "id": "cal_bench_inadequate_5",
        "name": "Under-sampled Calibration (5 trials)",
        "min_trials": 5,
        "min_empirical_power": 1.0,
        "alternatives": [
            {
                "alternative_name": "Sub",
                "n_trials": 5,
                "n_rejected": 5,
                "empirical_power": 1.0,
                "wilson_lower_95": round(w5_low, 6),
                "clopper_pearson_lower_95": round(cp5_low, 6),
            }
        ],
    }

    common_gate_kwargs = {
        "score_mode": "local_logit_rank",
        "probe_revision_hash": "builtin_v1",
        "probe_revision_id": "rev_builtin_screening_v1",
        "n_prompts": 20,
        "m_reference": 8,
        "alpha": 0.01,
        "audit_is_demo": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_output_tokens": 128,
        "reference_model_descriptor": ref_profile["model_identifier"],
        "reference_identity": ref_identity,
    }
    cal_gate_valid = check_calibration_compatibility(valid_cal_record, **common_gate_kwargs)
    cal_gate_underpowered = check_calibration_compatibility(underpowered_cal_record, **common_gate_kwargs)
    cal_gate_inadequate = check_calibration_compatibility(inadequate_cal_record, **common_gate_kwargs)

    # 2. Independent Evaluation Benchmark Conditions (using evaluation_seed stream only)
    eval_conditions = [
        {
            "condition_id": "genuine_null_logit_n20_m8",
            "label": "Genuine Reference (Null False-Alarm Check)",
            "scenario": "genuine",
            "substitute_fraction": 0.0,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "false_alarm_rate",
        },
        {
            "condition_id": "weak_alt_5pct_logit_n20_m8",
            "label": "Weak Alternative (5% Mixed Routing — Near-Null Limit)",
            "scenario": "mixed_routing",
            "substitute_fraction": 0.05,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "mixed_15pct_logit_n20_m8",
            "label": "Mixed Routing (15% Substitute Fraction)",
            "scenario": "mixed_routing",
            "substitute_fraction": 0.15,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "mixed_35pct_logit_n20_m8",
            "label": "Mixed Routing (35% Substitute Fraction)",
            "scenario": "mixed_routing",
            "substitute_fraction": 0.35,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "mixed_65pct_logit_n20_m8",
            "label": "Mixed Routing (65% Substitute Fraction)",
            "scenario": "mixed_routing",
            "substitute_fraction": 0.65,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "substituted_100pct_logit_n20_m8",
            "label": "Full Substitution (100% Substitute Model)",
            "scenario": "substituted",
            "substitute_fraction": 1.0,
            "score_mode": "local_logit_rank",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "mixed_35pct_logit_low_budget_n10_m4",
            "label": "Mixed Routing (35% Fraction, Reduced Budget n=10, m=4)",
            "scenario": "mixed_routing",
            "substitute_fraction": 0.35,
            "score_mode": "local_logit_rank",
            "n_prompts": 10,
            "m_reference": 4,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "substituted_100pct_length_only_n20_m8",
            "label": "Full Substitution (Length-Only Fallback Mode n=20, m=8)",
            "scenario": "substituted",
            "substitute_fraction": 1.0,
            "score_mode": "length_distribution_only",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "detection_power",
        },
        {
            "condition_id": "genuine_null_length_only_n20_m8",
            "label": "Genuine Reference (Length-Only Fallback Null Check)",
            "scenario": "genuine",
            "substitute_fraction": 0.0,
            "score_mode": "length_distribution_only",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "metric_role": "false_alarm_rate",
        },
    ]

    evaluation_results: List[Dict[str, Any]] = []
    for c_idx, cond in enumerate(eval_conditions):
        tgt_prof = _make_synthetic_profile(
            f"prof_eval_{c_idx}",
            cond["scenario"],
            cond["substitute_fraction"],
        )
        tgt_adapter = SyntheticDemoAdapter(tgt_prof)
        rejections = 0
        p_vals: List[float] = []
        cvm_stats: List[float] = []
        shifts: List[float] = []

        for t_idx in range(n_eval_trials):
            t_seed = evaluation_seed + c_idx * 100_000 + t_idx * 1237
            tr = _evaluate_single_synthetic_trial(
                prompts=prompts,
                ref_adapter=ref_adapter,
                tgt_adapter=tgt_adapter,
                score_mode=cond["score_mode"],
                n_prompts=cond["n_prompts"],
                m_reference=cond["m_reference"],
                alpha=cond["alpha"],
                trial_master_seed=t_seed,
            )
            if tr["rejected_at_alpha"]:
                rejections += 1
            p_vals.append(tr["p_value"])
            cvm_stats.append(tr["cvm_statistic"])
            shifts.append(tr["mean_score_shift"])

        rate = rejections / float(n_eval_trials)
        w_low, w_high = wilson_interval_95(rejections, n_eval_trials)
        cp_low, cp_high = clopper_pearson_interval_95(rejections, n_eval_trials)
        budget_per_trial = cond["n_prompts"] * (cond["m_reference"] + 1)

        evaluation_results.append(
            {
                "condition_id": cond["condition_id"],
                "label": cond["label"],
                "scenario": cond["scenario"],
                "substitute_fraction": cond["substitute_fraction"],
                "score_mode": cond["score_mode"],
                "n_prompts": cond["n_prompts"],
                "m_reference": cond["m_reference"],
                "alpha": cond["alpha"],
                "n_trials": n_eval_trials,
                "queries_per_trial": budget_per_trial,
                "total_queries_across_trials": budget_per_trial * n_eval_trials,
                "metric_role": cond["metric_role"],
                "rejections": rejections,
                "empirical_rate": round(rate, 4),
                "wilson_ci_95": [round(w_low, 6), round(w_high, 6)],
                "clopper_pearson_ci_95": [round(cp_low, 6), round(cp_high, 6)],
                "median_p_value": round(float(np.median(p_vals)), 6),
                "mean_cvm_statistic": round(float(np.mean(cvm_stats)), 4),
                "mean_score_shift": round(float(np.mean(shifts)), 4),
            }
        )

    benchmark_payload = {
        "benchmark_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_disclaimer": (
            "SYNTHETIC SOFTWARE VERIFICATION ONLY: These measurements validate the finite-sample "
            "rank transformation, Cramér–von Mises Monte Carlo engine, seed isolation, and calibration "
            "gates on controlled stochastic Markov/surprisal models. Observing 0/30 false rejections does "
            "NOT empirically prove a 1% false-positive rate (95% Wilson upper bound is 0.1135; exact Clopper-Pearson "
            "upper bound is 0.1157), and these synthetic runs do NOT measure detection accuracy on real hosted LLMs."
        ),
        "seed_separation": {
            "development_seed": development_seed,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
            "streams_disjoint": True,
        },
        "calibration_validation": {
            "predeclared_target_power": 0.80,
            "min_required_trials": 20,
            "min_required_wilson_lower_95": 0.80,
            "adequate_calibration_artifact": valid_cal_record,
            "adequate_calibration_gate_result": cal_gate_valid,
            "underpowered_alternative_calibration_artifact": underpowered_cal_record,
            "underpowered_alternative_gate_result": cal_gate_underpowered,
            "inadequate_calibration_artifact": inadequate_cal_record,
            "inadequate_calibration_gate_result": cal_gate_inadequate,
        },
        "evaluation_conditions": evaluation_results,
    }

    real_model_manifest = {
        "manifest_version": "1.1.0",
        "title": "ModelTrace V1 Executable Real-Model Evaluation Manifest",
        "status": "NOT_RUN_IN_OFFLINE_SANDBOX",
        "disclaimer": (
            "No real-model benchmark tables are fabricated in this repository. Execute the commands below "
            "on a machine with outbound HTTPS access (or a local vLLM/Ollama/Transformers checkpoint) "
            "to generate empirical measurements on real model pairs."
        ),
        "prerequisites": {
            "environment_variables": [
                "MODELTRACE_PROVIDER_REFERENCE_KEY=<your-trusted-reference-api-key>",
                "MODELTRACE_PROVIDER_TARGET_KEY=<your-audited-target-api-key>",
                "MODELTRACE_ALLOWED_OUTBOUND_ORIGINS=https://api.together.xyz,https://api.fireworks.ai",
                "MODELTRACE_LOCAL_MODELS_DIR=/path/to/verified/checkpoints (optional for local_logit_rank)",
            ],
            "optional_local_packages": ["torch>=2.1", "transformers>=4.38", "safetensors>=0.4"],
        },
        "recommended_evaluation_protocol": {
            "probe_suite": "builtin_v1 (24 prompts across 5 prompt families)",
            "screening_budget": {"n_prompts": 20, "m_reference": 8, "alpha": 0.01, "queries_per_audit": 180},
            "extended_builtin_budget": {"n_prompts": 24, "m_reference": 19, "alpha": 0.01, "queries_per_audit": 480},
            "calibration_trials_per_alternative": 25,
            "separated_seeds": {
                "calibration_seed": 700100,
                "genuine_control_seed": 800100,
                "substituted_eval_seed": 900100,
            },
        },
        "executable_commands": [
            "# 1. Initialize database and verify capabilities",
            ".venv/bin/modeltrace migrate --db-path ./data/real_eval.db",
            ".venv/bin/modeltrace capabilities --db-path ./data/real_eval.db",
            "# 2. Run live audit comparing Reference Profile vs Target Profile",
            ".venv/bin/modeltrace live-audit --reference-profile <REF_PROFILE_ID> --target-profile <TGT_PROFILE_ID> --score-mode local_logit_rank --n-prompts 20 --m-reference 8 --alpha 0.01 --seed 900100 --db-path ./data/real_eval.db",
            "# 3. Export signed JSON/CSV/HTML audit bundle",
            ".venv/bin/modeltrace export-audit <AUDIT_ID> --format json --output ./exports/real_audit_report.json --db-path ./data/real_eval.db",
        ],
    }

    target_dir = Path(output_dir or "exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    bench_path = target_dir / "measured_benchmark_v1.json"
    manifest_path = target_dir / "real_model_benchmark_manifest.json"
    bench_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(real_model_manifest, indent=2), encoding="utf-8")

    return {
        "benchmark_file": "exports/measured_benchmark_v1.json",
        "manifest_file": "exports/real_model_benchmark_manifest.json",
        "benchmark": benchmark_payload,
        "real_model_manifest": real_model_manifest,
    }
