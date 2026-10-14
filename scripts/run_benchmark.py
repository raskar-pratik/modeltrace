#!/usr/bin/env python3
"""Standalone runner for the ModelTrace V1 Measured Synthetic Benchmark Suite."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltrace.services.benchmark import run_measured_synthetic_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the ModelTrace V1 measured synthetic benchmark suite with separated seeds."
    )
    parser.add_argument("--eval-trials", type=int, default=30, help="Trials per evaluation condition (default: 30)")
    parser.add_argument("--cal-trials", type=int, default=25, help="Trials per calibration alternative (default: 25)")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "exports"), help="Directory for JSON outputs")
    args = parser.parse_args()

    result = run_measured_synthetic_benchmark(
        n_eval_trials=args.eval_trials,
        n_cal_trials=args.cal_trials,
        output_dir=args.output_dir,
    )
    summary_rows = []
    for cond in result["benchmark"]["evaluation_conditions"]:
        summary_rows.append(
            {
                "condition_id": cond["condition_id"],
                "score_mode": cond["score_mode"],
                "budget": f"n={cond['n_prompts']}, m={cond['m_reference']} ({cond['queries_per_trial']} q/trial)",
                "trials": cond["n_trials"],
                "metric_role": cond["metric_role"],
                "rejections": cond["rejections"],
                "rate": cond["empirical_rate"],
                "wilson_ci_95": cond["wilson_ci_95"],
                "median_p": cond["median_p_value"],
            }
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "benchmark_file": result["benchmark_file"],
                "manifest_file": result["manifest_file"],
                "calibration_valid_compatible": result["benchmark"]["calibration_validation"][
                    "adequate_calibration_gate_result"
                ]["compatible"],
                "calibration_inadequate_compatible": result["benchmark"]["calibration_validation"][
                    "inadequate_calibration_gate_result"
                ]["compatible"],
                "conditions": summary_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
