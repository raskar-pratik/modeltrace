"""Command-line interface for ModelTrace (`python -m modeltrace.cli`)."""

import json
import sys
from typing import Optional

import click
import uvicorn

from modeltrace.config import get_env_settings
from modeltrace.db.migrations import run_migrations
from modeltrace.db.repository import Repository
from modeltrace.schemas import (
    AuditCreateRequest,
    DemoCalibrationCreateRequest,
    ScoreMode,
    SyntheticScenario,
)
from modeltrace.services.benchmark import run_measured_synthetic_benchmark
from modeltrace.services.exporter import ExportService
from modeltrace.services.worker import AuditWorkerService


@click.group()
def cli() -> None:
    """ModelTrace CLI — Know what your API is serving."""


@cli.command("migrate")
@click.option("--db-path", default=None, help="Path to SQLite database file.")
def migrate_cmd(db_path: Optional[str]) -> None:
    """Apply pending database schema migrations and seed baseline records."""
    env = get_env_settings()
    target_db = db_path or env.db_path
    applied = run_migrations(target_db)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "db_path": target_db,
                "applied_migrations": applied,
            },
            indent=2,
        )
    )


@cli.command("capabilities")
@click.option("--db-path", default=None, help="Path to SQLite database file.")
def capabilities_cmd(db_path: Optional[str]) -> None:
    """Inspect runtime capabilities, adapters, and security configuration."""
    from modeltrace.api.app import create_app

    app = create_app(db_path=db_path, start_worker=False)
    for route in app.routes:
        if getattr(route, "path", None) == "/api/capabilities":
            res = route.endpoint()
            click.echo(json.dumps(res, indent=2))
            return


@cli.command("demo-audit")
@click.option(
    "--scenario",
    type=click.Choice(["genuine", "substituted", "mixed_routing"]),
    default="substituted",
    help="Synthetic DEMO target scenario.",
)
@click.option(
    "--score-mode",
    type=click.Choice(["local_logit_rank", "length_distribution_only"]),
    default="local_logit_rank",
    help="Primary scoring mode.",
)
@click.option("--n-prompts", default=20, type=int, help="Number of prompts (default 20).")
@click.option("--m-reference", default=8, type=int, help="References per prompt (default 8).")
@click.option("--alpha", default=0.01, type=float, help="Significance threshold alpha.")
@click.option("--seed", default=424242, type=int, help="Master RNG seed.")
@click.option(
    "--with-calibration/--no-calibration",
    default=False,
    help="Generate and attach a compatible DEMO calibration artifact.",
)
@click.option("--export-format", type=click.Choice(["json", "csv", "html"]), default="json")
@click.option("--db-path", default=None, help="Path to SQLite database file.")
def demo_audit_cmd(
    scenario: str,
    score_mode: str,
    n_prompts: int,
    m_reference: int,
    alpha: float,
    seed: int,
    with_calibration: bool,
    export_format: str,
    db_path: Optional[str],
) -> None:
    """Execute a complete seeded stochastic DEMO audit and output the verdict and export metadata."""
    env = get_env_settings()
    repo = Repository(db_path=db_path or env.db_path, auto_migrate=True)
    worker = AuditWorkerService(repo=repo, start_background_loop=False)
    exporter = ExportService(repo=repo)

    scenario_to_profile = {
        "genuine": "prof_demo_target_genuine",
        "substituted": "prof_demo_target_substituted",
        "mixed_routing": "prof_demo_target_mixed",
    }
    target_profile_id = scenario_to_profile[scenario]
    cal_id: Optional[str] = None

    if with_calibration:
        cal = worker.run_demo_calibration(
            DemoCalibrationCreateRequest(
                name=f"DEMO Calibration ({score_mode}, n={n_prompts}, m={m_reference})",
                score_mode=ScoreMode(score_mode),
                n_prompts=n_prompts,
                m_reference=m_reference,
                alpha=alpha,
                n_trials=20,
                calibration_seed=seed + 101,
            )
        )
        cal_id = cal["id"]

    create_req = AuditCreateRequest(
        name=f"[DEMO] CLI Audit ({scenario})",
        reference_profile_id="prof_demo_reference",
        target_profile_id=target_profile_id,
        calibration_id=cal_id,
        score_mode=ScoreMode(score_mode),
        n_prompts=n_prompts,
        m_reference=m_reference,
        max_output_tokens=128,
        alpha=alpha,
        master_seed=seed,
        start_immediately=False,
    )
    audit = worker.create_audit(create_req)
    audit = worker.start_audit(audit["id"], run_synchronously=True)
    _, _, exp_rec = exporter.export_audit(audit["id"], fmt=export_format)

    output = {
        "audit_id": audit["id"],
        "is_demo": audit["is_demo"],
        "scenario": scenario,
        "score_mode": audit["score_mode"],
        "execution_state": audit["execution_state"],
        "verdict": audit["verdict"],
        "verdict_reason": audit["verdict_reason"],
        "cvm_statistic": audit["summary"]["primary_test"]["cvm_statistic"],
        "p_value": audit["summary"]["primary_test"]["p_value"],
        "alpha": audit["alpha"],
        "export": exp_rec,
    }
    click.echo(json.dumps(output, indent=2))


@cli.command("live-audit")
@click.option("--reference-profile", required=True, help="Reference profile ID.")
@click.option("--target-profile", required=True, help="Target profile ID.")
@click.option(
    "--score-mode",
    type=click.Choice(["local_logit_rank", "length_distribution_only"]),
    default="length_distribution_only",
)
@click.option("--n-prompts", default=20, type=int)
@click.option("--m-reference", default=8, type=int)
@click.option("--alpha", default=0.01, type=float)
@click.option("--seed", default=424242, type=int)
@click.option("--calibration-id", default=None)
@click.option("--db-path", default=None)
def live_audit_cmd(
    reference_profile: str,
    target_profile: str,
    score_mode: str,
    n_prompts: int,
    m_reference: int,
    alpha: float,
    seed: int,
    calibration_id: Optional[str],
    db_path: Optional[str],
) -> None:
    """Create and synchronously execute a live audit against configured profiles."""
    env = get_env_settings()
    repo = Repository(db_path=db_path or env.db_path, auto_migrate=True)
    worker = AuditWorkerService(repo=repo, start_background_loop=False)

    req = AuditCreateRequest(
        name="CLI Live Audit",
        reference_profile_id=reference_profile,
        target_profile_id=target_profile,
        calibration_id=calibration_id,
        score_mode=ScoreMode(score_mode),
        n_prompts=n_prompts,
        m_reference=m_reference,
        alpha=alpha,
        master_seed=seed,
        start_immediately=False,
    )
    audit = worker.create_audit(req)
    audit = worker.start_audit(audit["id"], run_synchronously=True)
    click.echo(
        json.dumps(
            {
                "audit_id": audit["id"],
                "execution_state": audit["execution_state"],
                "verdict": audit["verdict"],
                "verdict_reason": audit["verdict_reason"],
                "summary": audit["summary"],
            },
            indent=2,
        )
    )


@cli.command("resume-audit")
@click.argument("audit_id")
@click.option("--db-path", default=None)
def resume_audit_cmd(audit_id: str, db_path: Optional[str]) -> None:
    """Explicitly resume an INTERRUPTED audit synchronously."""
    env = get_env_settings()
    repo = Repository(db_path=db_path or env.db_path, auto_migrate=True)
    worker = AuditWorkerService(repo=repo, start_background_loop=False)
    audit = worker.resume_audit(audit_id, run_synchronously=True)
    click.echo(
        json.dumps(
            {
                "audit_id": audit["id"],
                "execution_state": audit["execution_state"],
                "verdict": audit["verdict"],
                "resume_count": audit["resume_count"],
                "uncertain_prior_attempts": audit["uncertain_prior_attempts"],
            },
            indent=2,
        )
    )


@cli.command("export-audit")
@click.argument("audit_id")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "html"]), default="json")
@click.option("--output", default=None, help="Optional output file path.")
@click.option("--db-path", default=None)
def export_audit_cmd(
    audit_id: str, fmt: str, output: Optional[str], db_path: Optional[str]
) -> None:
    """Export a persisted audit report to JSON, CSV, or self-contained HTML."""
    env = get_env_settings()
    repo = Repository(db_path=db_path or env.db_path, auto_migrate=True)
    exporter = ExportService(repo=repo)
    _, _, exp_rec = exporter.export_audit(audit_id, fmt=fmt, output_path=output)
    click.echo(json.dumps(exp_rec, indent=2))


@cli.command("benchmark")
@click.option("--eval-trials", default=30, type=int, help="Evaluation trials per condition.")
@click.option("--cal-trials", default=25, type=int, help="Calibration trials per alternative.")
@click.option("--output-dir", default="exports", help="Directory for benchmark JSON outputs.")
def benchmark_cmd(eval_trials: int, cal_trials: int, output_dir: str) -> None:
    """Execute the separated-seed synthetic benchmark suite and generate real-model manifest."""
    res = run_measured_synthetic_benchmark(
        n_eval_trials=eval_trials,
        n_cal_trials=cal_trials,
        output_dir=output_dir,
    )
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "benchmark_file": res["benchmark_file"],
                "manifest_file": res["manifest_file"],
                "conditions_evaluated": len(res["benchmark"]["evaluation_conditions"]),
            },
            indent=2,
        )
    )


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host (defaults to 127.0.0.1).")
@click.option("--port", default=8000, type=int, help="Bind port (defaults to 8000).")
def serve_cmd(host: str, port: int) -> None:
    """Launch the localhost ModelTrace FastAPI server."""
    uvicorn.run("modeltrace.api.app:app", host=host, port=port, log_level="info")


def main() -> None:
    cli(prog_name="modeltrace")


if __name__ == "__main__":
    main()
