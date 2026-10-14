"""Tests for finite-sample tie-breaking rank transformation, CvM Monte Carlo test, Holm correction, bootstrap CI, Wilson & Clopper-Pearson intervals, strict calibration validation, and separated-seed synthetic benchmarks."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats

from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AuditVerdict,
    CalibrationAlternativeResult,
    CalibrationImportRequest,
    ExecutionState,
    ScoreMode,
)
from modeltrace.services.benchmark import run_measured_synthetic_benchmark
from modeltrace.services.stats import (
    STAT_ENGINE_METHOD,
    STAT_ENGINE_VERSION,
    build_reference_scorer_identity,
    check_calibration_compatibility,
    clopper_pearson_interval_95,
    compute_bootstrap_uncertainty,
    compute_cvm_statistic,
    compute_prompt_rank_z,
    derive_audit_seeds,
    evaluate_verdict,
    holm_step_down,
    run_cvm_monte_carlo_test,
    wilson_interval_95,
    wilson_lower_bound_95,
)


class TestStatisticsAndVerdicts(unittest.TestCase):
    def test_seed_stream_isolation(self) -> None:
        s1 = derive_audit_seeds(424242)
        s2 = derive_audit_seeds(424242)
        self.assertEqual(s1, s2)
        child_vals = [
            s1["reference_seed"],
            s1["target_seed"],
            s1["tie_break_seed"],
            s1["mc_null_seed"],
            s1["bootstrap_seed"],
        ]
        self.assertEqual(len(set(child_vals)), 5)

    def test_exact_uniformity_under_all_ties_and_continuous_null(self) -> None:
        rng = np.random.default_rng(20261014)
        z_ties = []
        for _ in range(400):
            res = compute_prompt_rank_z(
                reference_scores=[10.0] * 8,
                target_score=10.0,
                score_mode=ScoreMode.LENGTH_DISTRIBUTION_ONLY.value,
                rng=rng,
            )
            self.assertEqual(res["a_below"], 0)
            self.assertEqual(res["b_ties"], 8)
            self.assertGreaterEqual(res["z_uniform"], 0.0)
            self.assertLess(res["z_uniform"], 1.0)
            z_ties.append(res["z_uniform"])

        ks_stat, ks_p = scipy_stats.kstest(z_ties, "uniform")
        self.assertGreater(ks_p, 0.05, f"All-ties z failed uniformity KS test: p={ks_p}")

        mc_null = run_cvm_monte_carlo_test(z_ties[:20], mc_null_seed=999, n_simulations=9999)
        self.assertGreater(mc_null["p_value"], 0.01)

    def test_cvm_detects_shifted_distribution_and_enforces_min_b(self) -> None:
        with self.assertRaises(ValueError):
            run_cvm_monte_carlo_test([0.5] * 10, mc_null_seed=1, n_simulations=500)

        rng = np.random.default_rng(7777)
        z_shifted = []
        for _ in range(20):
            refs = rng.normal(loc=1.0, scale=0.3, size=8).tolist()
            tgt = float(rng.normal(loc=2.5, scale=0.3))
            rz = compute_prompt_rank_z(
                reference_scores=refs,
                target_score=tgt,
                score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
                rng=rng,
            )
            z_shifted.append(rz["z_uniform"])

        mc_shift = run_cvm_monte_carlo_test(z_shifted, mc_null_seed=12345, n_simulations=9999)
        self.assertLessEqual(mc_shift["p_value"], 0.005)
        self.assertGreater(mc_shift["cvm_statistic"], 1.0)

    def test_holm_step_down_and_bootstrap_uncertainty(self) -> None:
        raw_tests = [
            {"name": "t1", "raw_p_value": 0.002},
            {"name": "t2", "raw_p_value": 0.008},
            {"name": "t3", "raw_p_value": 0.040},
        ]
        corrected = holm_step_down(raw_tests, alpha=0.01)
        self.assertTrue(corrected[0]["rejected_at_alpha"])
        self.assertAlmostEqual(corrected[0]["holm_adjusted_p_value"], 0.006)
        self.assertFalse(corrected[1]["rejected_at_alpha"])
        self.assertFalse(corrected[2]["rejected_at_alpha"])

        prompt_records = [
            {"family": "code", "score_shift": 0.5, "z_uniform": 0.8},
            {"family": "code", "score_shift": 0.6, "z_uniform": 0.85},
            {"family": "reasoning", "score_shift": 0.4, "z_uniform": 0.75},
            {"family": "reasoning", "score_shift": 0.7, "z_uniform": 0.9},
        ]
        boot = compute_bootstrap_uncertainty(prompt_records, bootstrap_seed=42, n_replicates=500)
        self.assertAlmostEqual(boot["mean_score_shift"], 0.55)
        self.assertLessEqual(boot["prompt_bootstrap_ci_95"][0], 0.55)
        self.assertGreaterEqual(boot["prompt_bootstrap_ci_95"][1], 0.55)
        self.assertIn("code", boot["by_family_descriptive"])

    def test_verdict_separation_and_calibration_rules(self) -> None:
        # 1. Incomplete execution state -> INCONCLUSIVE even if p_value is tiny
        v1, _ = evaluate_verdict(
            execution_state=ExecutionState.INTERRUPTED.value,
            completed_prompts=19,
            expected_n_prompts=20,
            all_prompts_have_full_samples=False,
            p_value=0.0001,
            alpha=0.01,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            calibration_check={"compatible": True},
            is_demo=False,
        )
        self.assertEqual(v1, AuditVerdict.INCONCLUSIVE.value)

        # 2. Completed + p <= alpha -> INCONSISTENT
        v2, _ = evaluate_verdict(
            execution_state=ExecutionState.COMPLETED.value,
            completed_prompts=20,
            expected_n_prompts=20,
            all_prompts_have_full_samples=True,
            p_value=0.003,
            alpha=0.01,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            calibration_check={"compatible": False, "reason": "none"},
            is_demo=False,
        )
        self.assertEqual(v2, AuditVerdict.INCONSISTENT.value)

        # 3. Completed + p > alpha without compatible calibration -> INCONCLUSIVE
        cal_missing = check_calibration_compatibility(
            calibration=None,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            probe_revision_hash="hash_a",
            n_prompts=20,
            m_reference=8,
            alpha=0.01,
            audit_is_demo=False,
        )
        v3, _ = evaluate_verdict(
            execution_state=ExecutionState.COMPLETED.value,
            completed_prompts=20,
            expected_n_prompts=20,
            all_prompts_have_full_samples=True,
            p_value=0.42,
            alpha=0.01,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            calibration_check=cal_missing,
            is_demo=False,
        )
        self.assertEqual(v3, AuditVerdict.INCONCLUSIVE.value)

        # 4. Completed + p > alpha + compatible calibration (24/25 -> empirical_power=0.96, wilson_lower_95=0.8046 >= 0.80) -> CONSISTENT_WITHIN_TESTED_LIMITS
        ref_prof = {
            "adapter_type": "local_transformers",
            "model_identifier": "llama-3-8b-instruct",
            "checkpoint_path": "/models/llama-3-8b",
            "checkpoint_revision": "rev_abc123",
            "tokenizer_path": "/models/llama-3-8b",
            "chat_template": "<|user|>\n{prompt}\n<|assistant|>\n",
            "precision": "float16",
            "device": "cpu",
        }
        ref_identity = build_reference_scorer_identity(ref_prof, ref_prof)
        w24_25 = wilson_lower_bound_95(24, 25)
        self.assertGreaterEqual(w24_25, 0.80)
        valid_cal = {
            "id": "cal_valid",
            "name": "Valid Calibration",
            "score_mode": ScoreMode.LOCAL_LOGIT_RANK.value,
            "probe_revision_id": "rev_builtin_screening_v1",
            "probe_revision_hash": "hash_a",
            "n_prompts": 20,
            "m_reference": 8,
            "max_output_tokens": 128,
            "alpha": 0.01,
            "temperature": 0.7,
            "top_p": 0.95,
            "reference_model_descriptor": "llama-3-8b-instruct",
            "reference_identity": ref_identity,
            "stat_engine_method": STAT_ENGINE_METHOD,
            "stat_engine_version": STAT_ENGINE_VERSION,
            "is_demo": False,
            "min_trials": 25,
            "min_empirical_power": 0.96,
            "alternatives": [
                {
                    "alternative_name": "Int4-Sub",
                    "n_trials": 25,
                    "n_rejected": 24,
                    "empirical_power": 0.96,
                    "wilson_lower_95": round(w24_25, 6),
                }
            ],
        }
        cal_ok = check_calibration_compatibility(
            calibration=valid_cal,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            probe_revision_hash="hash_a",
            probe_revision_id="rev_builtin_screening_v1",
            n_prompts=20,
            m_reference=8,
            alpha=0.01,
            audit_is_demo=False,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=128,
            reference_model_descriptor="llama-3-8b-instruct",
            reference_identity=ref_identity,
        )
        self.assertTrue(cal_ok["compatible"], cal_ok.get("reason"))
        v4, reason4 = evaluate_verdict(
            execution_state=ExecutionState.COMPLETED.value,
            completed_prompts=20,
            expected_n_prompts=20,
            all_prompts_have_full_samples=True,
            p_value=0.42,
            alpha=0.01,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            calibration_check=cal_ok,
            is_demo=False,
        )
        self.assertEqual(v4, AuditVerdict.CONSISTENT_WITHIN_TESTED_LIMITS.value)
        self.assertIn("not proof of weights", reason4)

        # 5. Verify that 23/25 (point estimate 0.92 >= 0.80, but 95% Wilson lower bound 0.7503 < 0.80) is REJECTED by the >= 0.80 lower-bound policy
        w23_25 = wilson_lower_bound_95(23, 25)
        self.assertLess(w23_25, 0.80)
        cal_23_25 = dict(
            valid_cal,
            min_empirical_power=0.92,
            alternatives=[
                {
                    "alternative_name": "Int4-Sub",
                    "n_trials": 25,
                    "n_rejected": 23,
                    "empirical_power": 0.92,
                    "wilson_lower_95": round(w23_25, 6),
                }
            ],
        )
        cal_under_lb = check_calibration_compatibility(
            calibration=cal_23_25,
            score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
            probe_revision_hash="hash_a",
            probe_revision_id="rev_builtin_screening_v1",
            n_prompts=20,
            m_reference=8,
            alpha=0.01,
            audit_is_demo=False,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=128,
            reference_model_descriptor="llama-3-8b-instruct",
            reference_identity=ref_identity,
        )
        self.assertFalse(cal_under_lb["compatible"])
        self.assertIn("Wilson lower confidence bound", cal_under_lb["reason"])

    def test_forged_calibration_bounds_and_exhaustive_mismatch_regressions(self) -> None:
        # Verify exact Wilson and Clopper-Pearson values for n_trials=20, n_rejected=16
        w_low, w_high = wilson_interval_95(16, 20)
        cp_low, cp_high = clopper_pearson_interval_95(16, 20)
        self.assertAlmostEqual(w_low, 0.5839826, places=6)
        self.assertAlmostEqual(cp_low, 0.563386, places=5)
        self.assertNotAlmostEqual(w_low, cp_low, places=3)

        # 1. Forged wilson_lower_95=0.99 on n_trials=20, n_rejected=16 MUST fail import AND compatibility check
        with tempfile.TemporaryDirectory() as tmp:
            repo = Repository(str(Path(tmp) / "cal_test.db"))
            default_rev = repo.get_default_probe_revision()
            with self.assertRaises(ValueError) as ctx:
                repo.import_calibration(
                    CalibrationImportRequest(
                        name="Forged Wilson Calibration",
                        score_mode=ScoreMode.LOCAL_LOGIT_RANK,
                        probe_revision_id=default_rev["id"],
                        probe_revision_hash=default_rev["content_hash"],
                        n_prompts=20,
                        m_reference=8,
                        max_output_tokens=128,
                        alpha=0.01,
                        temperature=0.7,
                        top_p=0.95,
                        reference_model_descriptor="demo-llama-3-8b-instruct-fp16",
                        is_demo=True,
                        alternatives=[
                            CalibrationAlternativeResult(
                                alternative_name="Forged-Alt",
                                n_trials=20,
                                n_rejected=16,
                                empirical_power=0.80,
                                wilson_lower_95=0.99,
                            )
                        ],
                    )
                )
            self.assertIn("forged or inconsistent", str(ctx.exception))

            # 2. Honest wilson_lower_95=0.583983 on 16/20 imports into DB, but fails compatibility gate (< 0.80)
            honest_16_20 = repo.import_calibration(
                CalibrationImportRequest(
                    name="Marginal 16/20 Calibration",
                    score_mode=ScoreMode.LOCAL_LOGIT_RANK,
                    probe_revision_id=default_rev["id"],
                    probe_revision_hash=default_rev["content_hash"],
                    n_prompts=20,
                    m_reference=8,
                    max_output_tokens=128,
                    alpha=0.01,
                    temperature=0.7,
                    top_p=0.95,
                    reference_model_descriptor="demo-llama-3-8b-instruct-fp16",
                    is_demo=True,
                    alternatives=[
                        CalibrationAlternativeResult(
                            alternative_name="Marginal-Alt",
                            n_trials=20,
                            n_rejected=16,
                            empirical_power=0.80,
                            wilson_lower_95=round(w_low, 6),
                        )
                    ],
                )
            )
            gate_16_20 = check_calibration_compatibility(
                calibration=honest_16_20,
                score_mode=ScoreMode.LOCAL_LOGIT_RANK.value,
                probe_revision_hash=default_rev["content_hash"],
                n_prompts=20,
                m_reference=8,
                alpha=0.01,
                audit_is_demo=True,
            )
            self.assertFalse(gate_16_20["compatible"])
            self.assertIn("Wilson lower confidence bound", gate_16_20["reason"])

        # 3. Exhaustive negative regressions for each configuration mismatch
        ref_prof = {
            "adapter_type": "local_transformers",
            "model_identifier": "llama-3-8b-instruct",
            "base_url": "",
            "checkpoint_path": "/models/llama-3-8b",
            "checkpoint_revision": "rev_1",
            "tokenizer_path": "/models/llama-3-8b-tok",
            "chat_template": "<|user|>\n{prompt}\n<|assistant|>\n",
            "precision": "float16",
            "device": "cpu",
        }
        ref_id = build_reference_scorer_identity(ref_prof, ref_prof)
        w24_25 = wilson_lower_bound_95(24, 25)
        base_cal = {
            "id": "cal_base",
            "name": "Base Valid Calibration",
            "score_mode": ScoreMode.LOCAL_LOGIT_RANK.value,
            "probe_revision_id": "rev_1",
            "probe_revision_hash": "hash_1",
            "n_prompts": 20,
            "m_reference": 8,
            "max_output_tokens": 128,
            "alpha": 0.01,
            "temperature": 0.7,
            "top_p": 0.95,
            "reference_model_descriptor": "llama-3-8b-instruct",
            "reference_identity": ref_id,
            "stat_engine_method": STAT_ENGINE_METHOD,
            "stat_engine_version": STAT_ENGINE_VERSION,
            "is_demo": False,
            "min_trials": 25,
            "min_empirical_power": 0.96,
            "alternatives": [
                {
                    "alternative_name": "Sub",
                    "n_trials": 25,
                    "n_rejected": 24,
                    "empirical_power": 0.96,
                    "wilson_lower_95": round(w24_25, 6),
                }
            ],
        }
        base_kwargs = {
            "score_mode": ScoreMode.LOCAL_LOGIT_RANK.value,
            "probe_revision_hash": "hash_1",
            "probe_revision_id": "rev_1",
            "n_prompts": 20,
            "m_reference": 8,
            "alpha": 0.01,
            "audit_is_demo": False,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 128,
            "reference_model_descriptor": "llama-3-8b-instruct",
            "reference_identity": ref_id,
        }
        self.assertTrue(check_calibration_compatibility(base_cal, **base_kwargs)["compatible"])

        # Test each mismatch individually (including higher n_prompts=30 and higher m_reference=12)
        mismatch_overrides = [
            {"score_mode": ScoreMode.LENGTH_DISTRIBUTION_ONLY.value},
            {"probe_revision_hash": "hash_different"},
            {"probe_revision_id": "rev_different"},
            {"n_prompts": 10},
            {"n_prompts": 30},  # Higher budget without validated coverage must fail
            {"m_reference": 4},
            {"m_reference": 12},  # Higher reference count without validated coverage must fail
            {"alpha": 0.05},
            {"temperature": 0.2},
            {"top_p": 0.80},
            {"max_output_tokens": 64},
            {"audit_is_demo": True},
            {"reference_model_descriptor": "mistral-7b-instruct"},
            {"stat_engine_method": "ks_test"},
            {"stat_engine_version": "0.9.0"},
        ]
        for override in mismatch_overrides:
            kw = dict(base_kwargs, **override)
            res = check_calibration_compatibility(base_cal, **kw)
            self.assertFalse(res["compatible"], f"Expected incompatibility for override {override}")

        # Test each reference/scorer identity sub-field mismatch
        for id_field in (
            "reference_checkpoint_path",
            "reference_checkpoint_revision",
            "reference_tokenizer_path",
            "reference_chat_template",
            "reference_precision",
            "reference_device",
            "scorer_model_identifier",
            "scorer_chat_template",
        ):
            mutated_id = dict(ref_id, **{id_field: "mismatched_value"})
            kw = dict(base_kwargs, reference_identity=mutated_id)
            res = check_calibration_compatibility(base_cal, **kw)
            self.assertFalse(res["compatible"], f"Expected incompatibility for identity field {id_field}")

        # Test empty/malformed/non-finite artifacts
        for bad_cal in [
            dict(base_cal, alternatives=[]),
            dict(base_cal, alpha=float("nan")),
            dict(base_cal, alternatives=[{"alternative_name": "Bad", "n_trials": 20, "n_rejected": 25}]),
        ]:
            res = check_calibration_compatibility(bad_cal, **base_kwargs)
            self.assertFalse(res["compatible"])

    def test_separated_seed_benchmark_runner(self) -> None:
        with self.assertRaises(ValueError):
            run_measured_synthetic_benchmark(
                n_eval_trials=2, n_cal_trials=2, development_seed=10, calibration_seed=10, evaluation_seed=20
            )
        with tempfile.TemporaryDirectory() as tmp:
            res = run_measured_synthetic_benchmark(
                n_eval_trials=6,
                n_cal_trials=25,
                output_dir=tmp,
            )
            bench = res["benchmark"]
            self.assertTrue(bench["seed_separation"]["streams_disjoint"])
            self.assertEqual(bench["calibration_validation"]["min_required_wilson_lower_95"], 0.80)
            self.assertTrue(bench["calibration_validation"]["adequate_calibration_gate_result"]["compatible"])
            self.assertFalse(bench["calibration_validation"]["inadequate_calibration_gate_result"]["compatible"])
            self.assertEqual(len(bench["evaluation_conditions"]), 9)
            self.assertIn("clopper_pearson_ci_95", bench["evaluation_conditions"][0])


if __name__ == "__main__":
    unittest.main()
