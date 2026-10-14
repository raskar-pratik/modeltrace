# Architectural, Product & Release Decisions (`get-more-perspectives` Synthesis)

## 1. Stage 1 Virtual Advisory Panel Review (Architecture & Statistical Foundation)

**Decision Framed:** How should ModelTrace architect its Stage 1 backend, statistical verification engine, execution worker, and adapter boundaries to deliver scientifically defensible black-box/gray-box model API auditing without over-claiming identity proof or risking runaway query spend?

### Executive Summary & Panel Recommendation

- **Vote:** 5 Conditional Yes (Unanimous approval subject to strict statistical, security, and budget guardrails)
- **Consensus:** Build a single-user localhost modular monolith in Python (FastAPI + SQLite + single background worker) that enforces pre-frozen audit manifests, persists every sample attempt before network invocation, separates execution state from inferential verdict, and restricts positive consistency claims (`CONSISTENT_WITHIN_TESTED_LIMITS`) to runs backed by a compatible power calibration artifact.
- **Key Condition:** Never equate statistical non-rejection with proof of model weights or provider honesty, and never allow response-length fallback (`LENGTH_DISTRIBUTION_ONLY`) to inherit the sensitivity claims of full-vocabulary teacher-forcing logit-rank scoring (`LOCAL_LOGIT_RANK`).

### Individual Advisor Perspectives (Stage 1)

**The Operator (Execution & Reliability):**
Yes, with strict process boundaries. Heavy statistical simulations and local teacher-forcing scoring must never run inside FastAPI request handlers; use a single database-backed background worker thread with atomic state transitions (`CREATED -> RUNNING`) and pre-call `sample_attempts` persistence. On process restart, any unfinished job (`RUNNING` or `STARTING`) must immediately transition to `INTERRUPTED` rather than auto-retrying in the background, and explicit resume must reuse completed prompt samples while accounting conservatively for any `IN_FLIGHT` attempts whose external status is uncertain.

**The Skeptic (Statistical Validity & Confounders):**
Yes, only if we eliminate p-hacking and over-claiming. Repeated significance peeking or early stopping invalidates Type I error control; the audit manifest must freeze the probe revision, score mode, sampling parameters, $\alpha$, and independent RNG seeds before a single query is sent. Ties in finite-sample ranks must be broken via exact randomized uniformization $z_i = (a_i + J_i + U_i)/(m + 1)$ rather than ad-hoc tolerances, aggregated with Cramér–von Mises ($B \ge 9999$ Monte Carlo uniform null simulations), and every report must explicitly warn that chat templates, hidden system prompts, safety filters, and non-deterministic GPU kernels can trigger `INCONSISTENT` verdicts even when base weights match.

**The Visionary (Extension Boundaries):**
Yes, provided our adapter and scoring abstractions cleanly separate *generation* from *scoring*. By defining a uniform adapter contract (`SyntheticDemoAdapter`, `OpenAICompatibleAdapter`, `LocalTransformersAdapter`) and storing immutable probe revisions and calibration artifacts with SHA-256 content hashes, ModelTrace can start as an offline/localhost auditor today and later incorporate multi-checkpoint differential diagnosis or logprob-API adapters without changing the core CvM rank engine or database schema.

**The Customer Advocate (Usable Honest Verdicts):**
Yes, if verdicts are crystal clear to a practitioner who isn't a statistician. Users need a three-way verdict (`INCONSISTENT`, `INCONCLUSIVE`, `CONSISTENT_WITHIN_TESTED_LIMITS`) that is strictly decoupled from job execution state (`COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`). When an audit fails to reject the null without an attached power calibration artifact, returning `INCONCLUSIVE` (instead of a misleading green checkmark) protects users from false confidence while explaining exactly what calibration is required to claim consistency within tested limits.

**The Finance Mind (Query & Compute Budgets):**
Yes, with hard pre-flight caps. Querying a reference model $m=8$ times plus a target $1$ time across $n=20$ prompts costs $180$ completions per screening run, which scales quickly if retries loop on 429s or page loads trigger audits. We must require deliberate audit start, provide a zero-cost `POST /api/audits/preview` budget calculator, enforce hard per-audit caps on target attempts, reference attempts, total attempts (including retries), and wall-clock duration, and provide a seeded stochastic `DEMO` mode that exercises the full statistical and persistence pipeline at zero API cost.

## 2. Stage 3 Virtual Release Panel Review (Final V1 Release Readiness)

**Decision Framed:** Is ModelTrace V1 ready for release across statistical validity, security hardening, measured synthetic benchmarking, and clean-environment reproducibility, and what release gates must be enforced before shipping?

> *Scope Note:* This panel is structured engineering reasoning (`get-more-perspectives`) used to audit and resolve release gaps within this task—not an independent external audit, autonomous sub-agent execution, or substitute for measured test evidence.

### Individual Advisor Perspectives (Stage 3 Release Audit)

**The Operator (Reproducibility & Clean-Environment Execution):**
- *Finding:* A release is only reproducible if a fresh data directory (`--db-path`) initializes cleanly via migrations, runs the CLI and single-origin FastAPI server without hidden state, and passes automated CI checks without requiring paid API keys or GPU hardware.
- *Resolution Implemented:* Verified clean-environment database creation, added `.github/workflows/ci.yml` executing Python static checks, frontend compilation (`node scripts/build_frontend.cjs`), CLI smoke tests, controlled HTTP server tests, and headless Chromium E2E tests, and added `modeltrace benchmark` / `scripts/run_benchmark.py`.

**The Skeptic (Statistical Validity, Uncertainty Bounds & Security Hardening):**
- *Finding 1 (Binomial Calibration Uncertainty):* A calibration artifact claiming $1.00$ empirical power on only $5$ trials has wide binomial uncertainty (95% Wilson lower bound $0.5655$) and must not qualify an audit for `CONSISTENT_WITHIN_TESTED_LIMITS`. Furthermore, synthetic benchmark seeds must be strictly separated across `development_seed`, `calibration_seed`, and `evaluation_seed`, and weak alternatives (e.g., 5% mixed routing) must be reported honestly when they fail to reject at screening budgets.
- *Finding 2 (Export Sanitization):* Untrusted model responses or hostile audit/profile names could contain spreadsheet formulas (`=CMD|...`, `+SUM(...)`, `-2+3`, `@HYPERLINK(...)`) or raw `<script>` / `<img onerror=...>` payloads.
- *Resolution Implemented:* Added `wilson_interval_95()` and enforced `min_trials >= 20`, `empirical_power >= 0.80`, and `wilson_lower_95 >= 0.60` in `check_calibration_compatibility()`. Implemented `neutralize_csv_cell()` in `modeltrace/services/exporter.py`, escaped stored response previews in HTML reports via `html.escape`, and verified both with unit tests (`tests/test_security_and_redaction.py`).

**The Customer Advocate (Usability, Responsive Hierarchy & Honest Reporting):**
- *Finding:* Practitioners inspecting HTML/JSON reports after running an evidence retention purge need explicit confirmation that raw completions were intentionally purged while the frozen manifest and rank scores remain intact (`PURGED_OR_EMPTY_MANIFEST_PRESERVED` vs `FULL_RESPONSES_PRESENT`).
- *Resolution Implemented:* Added explicit `evidence_retention` status and stored response counts to `ExportService.build_export_bundle()` and rendered HTML reports, and verified before/after purge snapshot agreement in `tests/test_security_and_redaction.py`.

**The Finance Mind (Budget Protection & Real-Model Evaluation Manifest):**
- *Finding:* Ordinary CI and offline verification must never trigger paid API calls, and upstream `500`/`502`/`429` errors must be strictly bounded by `max_total_attempts`. For teams ready to run paid live evaluations later, we should provide an executable manifest with explicit query budgets (`180` queries for screening vs `480` queries for high-sensitivity runs).
- *Resolution Implemented:* Added HTTP `500`/`502` retry and attempt-cap tests in `tests/test_http_adapter_and_worker.py` and generated `exports/real_model_benchmark_manifest.json` with exact CLI commands and budget formulas.

**The Visionary (Future Boundaries & Non-Claims):**
- *Finding:* V1 must draw a bright line between behavioral auditing (supported today) and weight/quantization forensics or multi-tenant cloud hosting (out of scope for V1).
- *Resolution Implemented:* Documented explicit scientific and security boundaries across `README.md`, `STATISTICAL_METHOD.md`, `SECURITY.md`, and `VALIDATION_REPORT.md`.
