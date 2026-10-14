"""Reproducible JSON, CSV, and self-contained HTML report exporter for ModelTrace.

Adheres to:
- Strict secret redaction (never exports raw keys or unredacted headers).
- Neutralizes CSV formula injection (`=`, `+`, `-`, `@`, `\t`, `\r` prefixes on text cells) and escapes hostile HTML/script payloads (`html.escape`).
- Reproduces stored results deterministically from SQLite records, reports evidence retention status (`FULL_RESPONSES_PRESENT` vs `PURGED_MANIFEST_PRESERVED`), and records SHA-256 integrity hashes.
- HTML report follows the `minimalist-ui` specification: warm monochrome canvas (#F7F6F3 / #FFFFFF), charcoal text (#111111 / #787774), 1px solid #EAEAEA borders, muted pastel semantic badges (#FDEBEC, #E1F3FE, #EDF3EC, #FBF3DB), editorial serif + geometric sans + monospace font hierarchy, zero emojis, and zero heavy shadows.
"""

import csv
import html
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modeltrace.config import APP_NAME, APP_TAGLINE, canonical_json, get_env_settings, sha256_hex, utc_now_iso
from modeltrace.db.repository import Repository
from modeltrace.schemas import AuditVerdict, ScoreMode
from modeltrace.utils.security import redact_secrets

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_csv_cell(val: Any) -> Any:
    """Neutralize spreadsheet formula injection in string CSV cells while preserving numbers."""
    if val is None or isinstance(val, (int, float, bool)):
        return val
    text = str(val)
    stripped = text.lstrip(" ")
    if stripped and stripped.startswith(DANGEROUS_CSV_PREFIXES):
        return f"'{text}"
    return text


class ExportService:
    """Generates and persists JSON, CSV, and self-contained HTML exports for persisted audits."""

    def __init__(self, repo: Repository, exports_dir: Optional[str] = None) -> None:
        self.repo = repo
        env = get_env_settings()
        self.exports_dir = Path(exports_dir or env.exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def build_export_bundle(self, audit_id: str) -> Dict[str, Any]:
        audit = self.repo.get_audit(audit_id)
        if not audit:
            raise KeyError(f"Audit '{audit_id}' not found.")
        prompt_scores = self.repo.list_prompt_scores(audit_id)
        events = self.repo.list_audit_events(audit_id)
        attempts = self.repo.list_audit_attempts(audit_id)
        responses = self.repo.list_audit_responses(audit_id)
        calibration = (
            self.repo.get_calibration(audit["calibration_id"])
            if audit.get("calibration_id")
            else None
        )

        retention_status = (
            "FULL_RESPONSES_PRESENT"
            if len(responses) > 0
            else "PURGED_OR_EMPTY_MANIFEST_PRESERVED"
        )

        bundle = {
            "application": APP_NAME,
            "tagline": APP_TAGLINE,
            "exported_at_utc": utc_now_iso(),
            "is_demo": bool(audit["is_demo"]),
            "demo_banner": (
                "SYNTHETIC DEMO REPORT — Generated from seeded stochastic simulation, not live API traffic."
                if audit["is_demo"]
                else None
            ),
            "score_mode_scope": (
                "LENGTH DISTRIBUTION ONLY — Evaluates Unicode response length only; cannot verify token-level identity."
                if audit["score_mode"] == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value
                else "LOCAL_LOGIT_RANK — Teacher-forcing mean log vocabulary rank under reference logits."
            ),
            "epistemic_notice": (
                "Evidence concerns observable behavior under the frozen test configuration. "
                "It does not prove model weights, parameter count, precision, provider intent, or fraud."
            ),
            "evidence_retention": {
                "status": retention_status,
                "stored_response_count": len(responses),
                "prompt_score_count": len(prompt_scores),
            },
            "audit": redact_secrets(audit),
            "calibration": redact_secrets(calibration) if calibration else None,
            "prompt_scores": redact_secrets(prompt_scores),
            "stored_responses": redact_secrets(responses),
            "attempt_summary": {
                "total_attempts": len(attempts),
                "attempts": redact_secrets(attempts),
            },
            "events": redact_secrets(events),
        }
        bundle["export_integrity_sha256"] = sha256_hex(bundle)
        return bundle

    def export_audit(
        self,
        audit_id: str,
        fmt: str = "json",
        output_path: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Generate export in ('json', 'csv', 'html'), save to disk, and record export metadata."""
        fmt_clean = (fmt or "json").strip().lower()
        bundle = self.build_export_bundle(audit_id)

        if fmt_clean == "json":
            content = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False)
            media_type = "application/json"
            ext = "json"
        elif fmt_clean == "csv":
            content = self._render_csv(bundle)
            media_type = "text/csv"
            ext = "csv"
        elif fmt_clean == "html":
            content = self._render_html(bundle)
            media_type = "text/html"
            ext = "html"
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Choose 'json', 'csv', or 'html'.")

        content_bytes = content.encode("utf-8")
        content_hash = sha256_hex(content_bytes)

        if output_path:
            target_file = Path(output_path)
        else:
            target_file = self.exports_dir / f"{audit_id}_report.{ext}"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(content_bytes)

        exp_rec = self.repo.record_export(
            audit_id=audit_id,
            export_format=fmt_clean,
            file_path=str(target_file.resolve()),
            content_hash=content_hash,
            byte_size=len(content_bytes),
        )
        return content, media_type, exp_rec

    def _render_csv(self, bundle: Dict[str, Any]) -> str:
        audit = bundle["audit"]
        scores = bundle["prompt_scores"]
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            [
                "audit_id",
                "audit_name",
                "is_demo",
                "execution_state",
                "verdict",
                "score_mode",
                "prompt_index",
                "prompt_id",
                "family",
                "m_reference",
                "target_score",
                "mean_reference_score",
                "score_shift",
                "a_below",
                "b_ties",
                "j_draw",
                "u_draw",
                "z_uniform",
                "target_norm_ll_descriptive",
                "ref_mean_norm_ll_descriptive",
                "local_scoring_compute_ms",
            ]
        )
        for s in scores:
            ref_list = s.get("reference_scores") or []
            mean_ref = sum(ref_list) / len(ref_list) if ref_list else 0.0
            writer.writerow(
                [
                    neutralize_csv_cell(audit["id"]),
                    neutralize_csv_cell(audit["name"]),
                    "DEMO" if audit["is_demo"] else "LIVE",
                    neutralize_csv_cell(audit["execution_state"]),
                    neutralize_csv_cell(audit["verdict"]),
                    neutralize_csv_cell(s["score_mode"]),
                    int(s["prompt_index"]),
                    neutralize_csv_cell(s["prompt_id"]),
                    neutralize_csv_cell(s["family"]),
                    int(s["m_reference"]),
                    f"{float(s['target_score']):.6f}",
                    f"{mean_ref:.6f}",
                    f"{float(s['score_shift']):.6f}",
                    int(s["a_below"]),
                    int(s["b_ties"]),
                    int(s["j_draw"]),
                    f"{float(s['u_draw']):.6f}",
                    f"{float(s['z_uniform']):.6f}",
                    (
                        f"{float(s['target_norm_ll']):.6f}"
                        if s.get("target_norm_ll") is not None
                        else ""
                    ),
                    (
                        f"{float(s['ref_mean_norm_ll']):.6f}"
                        if s.get("ref_mean_norm_ll") is not None
                        else ""
                    ),
                    f"{float(s.get('local_scoring_compute_ms', 0.0)):.3f}",
                ]
            )
        return buf.getvalue()

    def _render_html(self, bundle: Dict[str, Any]) -> str:
        audit = bundle["audit"]
        summary = audit.get("summary") or {}
        primary = summary.get("primary_test") or {}
        uncertainty = summary.get("uncertainty") or {}
        cal_eval = summary.get("calibration_evaluation") or {}
        scores = bundle["prompt_scores"]
        responses = bundle.get("stored_responses") or []
        ev_retention = bundle.get("evidence_retention") or {}

        verdict = str(audit.get("verdict", "INCONCLUSIVE"))
        if verdict == AuditVerdict.INCONSISTENT.value:
            v_bg, v_fg = "#FDEBEC", "#9F2F2D"
        elif verdict == AuditVerdict.CONSISTENT_WITHIN_TESTED_LIMITS.value:
            v_bg, v_fg = "#EDF3EC", "#346538"
        else:
            v_bg, v_fg = "#FBF3DB", "#956400"

        demo_banner_html = ""
        if audit.get("is_demo"):
            demo_banner_html = """
            <div class="banner banner-demo">
                <span class="badge badge-blue">DEMO MODE</span>
                <span>Synthetic stochastic simulation report. Generated without external API keys or live model weights.</span>
            </div>
            """

        length_banner_html = ""
        if audit.get("score_mode") == ScoreMode.LENGTH_DISTRIBUTION_ONLY.value:
            length_banner_html = """
            <div class="banner banner-warn">
                <span class="badge badge-yellow">LENGTH DISTRIBUTION ONLY</span>
                <span>Evaluated using frozen Unicode response length fallback. Does not verify token-level identity or local logit-rank sensitivity.</span>
            </div>
            """

        rows_html_list = []
        for s in scores:
            refs = s.get("reference_scores") or []
            mean_ref = sum(refs) / len(refs) if refs else 0.0
            rows_html_list.append(
                f"""
                <tr>
                    <td class="mono">{html.escape(str(s['prompt_id']))}</td>
                    <td><span class="badge badge-neutral">{html.escape(str(s['family']))}</span></td>
                    <td class="mono num">{float(s['target_score']):.4f}</td>
                    <td class="mono num">{mean_ref:.4f}</td>
                    <td class="mono num">{float(s['score_shift']):+.4f}</td>
                    <td class="mono num">{int(s['a_below'])} / {int(s['b_ties'])}</td>
                    <td class="mono num">{float(s['z_uniform']):.4f}</td>
                </tr>
                """
            )
        rows_html = "\n".join(rows_html_list) or '<tr><td colspan="7">No prompt scores recorded.</td></tr>'

        resp_rows_list = []
        for r in responses[:20]:
            resp_rows_list.append(
                f"""
                <tr>
                    <td class="mono">{html.escape(str(r['prompt_id']))} ({html.escape(str(r['role']))} #{int(r['sample_index'])})</td>
                    <td class="mono">{html.escape(str(r['response_text'])[:160])}</td>
                    <td class="mono num">{float(r['raw_score']):.4f}</td>
                </tr>
                """
            )
        resp_rows_html = (
            "\n".join(resp_rows_list)
            or '<tr><td colspan="3">Raw responses not present (purged under retention policy while preserving manifest and rank statistics).</td></tr>'
        )

        cvm_val = (
            f"{float(primary['cvm_statistic']):.4f}"
            if "cvm_statistic" in primary
            else "N/A"
        )
        p_val = f"{float(primary['p_value']):.5f}" if "p_value" in primary else "N/A"
        ci_p = uncertainty.get("prompt_bootstrap_ci_95") or [0.0, 0.0]

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(APP_NAME)} Audit Report — {html.escape(str(audit['id']))}</title>
<style>
  :root {{
    --bg-canvas: #F7F6F3;
    --bg-surface: #FFFFFF;
    --text-primary: #111111;
    --text-secondary: #787774;
    --border-subtle: #EAEAEA;
    --font-sans: 'SF Pro Display', 'Geist Sans', 'Helvetica Neue', 'Switzer', sans-serif;
    --font-serif: 'Lyon Text', 'Newsreader', 'Playfair Display', 'Instrument Serif', serif;
    --font-mono: 'Geist Mono', 'SF Mono', 'JetBrains Mono', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-canvas);
    color: var(--text-primary);
    font-family: var(--font-sans);
    line-height: 1.6;
    padding: 48px 24px;
  }}
  .container {{
    max-width: 960px;
    margin: 0 auto;
  }}
  header {{
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 28px;
    margin-bottom: 32px;
  }}
  .eyebrow {{
    font-family: var(--font-mono);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }}
  h1 {{
    font-family: var(--font-serif);
    font-size: 36px;
    line-height: 1.1;
    letter-spacing: -0.025em;
    font-weight: 600;
    margin-bottom: 10px;
  }}
  .tagline {{
    color: var(--text-secondary);
    font-size: 15px;
  }}
  .banner {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
  }}
  .bento-grid {{
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }}
  .card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 24px;
  }}
  .col-8 {{ grid-column: span 8; }}
  .col-4 {{ grid-column: span 4; }}
  .col-6 {{ grid-column: span 6; }}
  .col-12 {{ grid-column: span 12; }}
  .card-label {{
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }}
  .metric-large {{
    font-family: var(--font-serif);
    font-size: 28px;
    line-height: 1.15;
    margin-bottom: 8px;
  }}
  .badge {{
    display: inline-block;
    border-radius: 9999px;
    padding: 3px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }}
  .badge-red {{ background: #FDEBEC; color: #9F2F2D; }}
  .badge-blue {{ background: #E1F3FE; color: #1F6C9F; }}
  .badge-green {{ background: #EDF3EC; color: #346538; }}
  .badge-yellow {{ background: #FBF3DB; color: #956400; }}
  .badge-neutral {{ background: #F7F6F3; color: #2F3437; border: 1px solid #EAEAEA; }}
  .mono {{ font-family: var(--font-mono); font-size: 13px; }}
  .num {{ text-align: right; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  th, td {{
    border-bottom: 1px solid var(--border-subtle);
    padding: 10px 12px;
    text-align: left;
  }}
  th {{
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    background: #F9F9F8;
  }}
  .epistemic-box {{
    font-size: 13px;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-subtle);
    padding-top: 16px;
    margin-top: 24px;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="eyebrow">{html.escape(APP_NAME)} — {html.escape(APP_TAGLINE)}</div>
    <h1>{html.escape(str(audit['name']))}</h1>
    <div class="tagline">Audit ID: <span class="mono">{html.escape(str(audit['id']))}</span> &middot; Manifest SHA-256: <span class="mono">{html.escape(str(audit['manifest_hash'])[:16])}...</span></div>
  </header>

  {demo_banner_html}
  {length_banner_html}

  <div class="bento-grid">
    <div class="card col-8">
      <div class="card-label">Statistical Verdict &amp; Execution State</div>
      <div style="margin-bottom: 12px;">
        <span class="badge" style="background:{v_bg}; color:{v_fg};">{html.escape(verdict)}</span>
        <span class="badge badge-neutral" style="margin-left: 8px;">STATE: {html.escape(str(audit['execution_state']))}</span>
        <span class="badge badge-neutral" style="margin-left: 8px;">MODE: {html.escape(str(audit['score_mode']))}</span>
      </div>
      <p style="font-size: 14px; color: #2F3437;">{html.escape(str(audit.get('verdict_reason', '')))}</p>
    </div>

    <div class="card col-4">
      <div class="card-label">Cram&eacute;r&ndash;von Mises Null Test</div>
      <div class="metric-large">p = {html.escape(p_val)}</div>
      <div class="mono" style="color: var(--text-secondary);">
        W&sup2; = {html.escape(cvm_val)} &middot; &alpha; = {float(audit['alpha']):.4f}<br>
        B = {int(audit['mc_simulations'])} MC runs &middot; n = {int(audit['n_prompts'])}, m = {int(audit['m_reference'])}
      </div>
    </div>

    <div class="card col-6">
      <div class="card-label">Score-Shift Uncertainty (Bootstrap 95% CI)</div>
      <div class="mono">
        Mean Shift (Target - Ref): {float(uncertainty.get('mean_score_shift', 0.0)):+.4f}<br>
        Prompt-Level 95% CI: [{float(ci_p[0]):+.4f}, {float(ci_p[1]):+.4f}]<br>
        Mean Rank Deviation (z - 0.5): {float(uncertainty.get('mean_z_shift_from_half', 0.0)):+.4f}
      </div>
    </div>

    <div class="card col-6">
      <div class="card-label">Calibration, Retention &amp; Compute Provenance</div>
      <div class="mono">
        Calibration Compatible: {html.escape(str(cal_eval.get('compatible', False)))}<br>
        Evidence Retention: {html.escape(str(ev_retention.get('status', 'UNKNOWN')))} ({int(ev_retention.get('stored_response_count', 0))} responses)<br>
        Local Scoring Compute: {float(audit.get('local_scoring_compute_ms', 0.0)):.2f} ms<br>
        Report Integrity SHA-256: {html.escape(str(bundle['export_integrity_sha256'])[:20])}...
      </div>
    </div>

    <div class="card col-12">
      <div class="card-label">Per-Prompt Finite-Sample Rank Breakdown (z = (a + J + U) / (m + 1))</div>
      <table>
        <thead>
          <tr>
            <th>Prompt ID</th>
            <th>Family</th>
            <th class="num">Target Score</th>
            <th class="num">Ref Mean Score</th>
            <th class="num">Shift</th>
            <th class="num">a_below / b_ties</th>
            <th class="num">Uniform z_i</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <div class="card col-12">
      <div class="card-label">Stored Response Evidence Snapshot (Escaped Preview)</div>
      <table>
        <thead>
          <tr>
            <th>Prompt &amp; Sample</th>
            <th>Escaped Response Text</th>
            <th class="num">Score</th>
          </tr>
        </thead>
        <tbody>
          {resp_rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <div class="epistemic-box">
    <strong>Epistemic &amp; Confounder Scope:</strong> {html.escape(str(bundle['epistemic_notice']))}
    Differences in chat templates, hidden system prompts, safety filters, or GPU kernel reductions can trigger INCONSISTENT verdicts even when base model weights match.
  </div>
</div>
</body>
</html>
"""
