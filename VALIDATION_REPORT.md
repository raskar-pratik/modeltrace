# ModelTrace 1.0.0 — release verification

**Release status: verified single-user local product.** The experimental preview label has been removed after completing the release verification on suitable GitHub runners.

Validated source: `7ebbcf4a8c22da6b4d37a876c826fac75e0d5654`.

[Release CI: all four jobs passed](https://github.com/raskar-pratik/modeltrace/actions/runs/38584463897), 14 October 2026. The final evidence-only documentation commit preserves the exact tested application, tests, build configuration, and dependency files. Their SHA-256 fingerprints are in `release-evidence/github-final-verification.json`.

## Executed release checks

| Check | Result | Evidence |
|---|---|---|
| Linux Python stack and production frontend | PASS | Locked Python install, pip check, npm ci, actual tsc negative control and Vite build |
| Connected Chrome browser workflow | PASS | Core/browser suite: **31 passed, 1 skipped**; eight screens, synthetic audit, reports, retention, responsive layouts and loaded SVG mark |
| Optional HF check in core job | SKIPPED THERE | Executed as an actual integration test in the dedicated HF job below |
| Native Windows setup | PASS | Real PowerShell setup, pip check, frontend build, migration, launcher and local HTTP delivery |
| Native Windows failure guard | PASS | Targeted suite: **5 passed, 1 skipped**; the intentionally failed native process stops execution. The skipped POSIX shell test runs on Linux |
| Actual local HuggingFace adapter | PASS | **1 integration test passed** using torch 2.5.1+cpu / transformers 4.46.3 against a locally constructed small causal checkpoint; generation and teacher-forced scoring executed |
| Docker image | PASS | Multi-stage image build, container health, real JS/CSS assets and bounded synthetic demo |
| Seeded synthetic software benchmarks | PASS | Separated development/calibration/evaluation seeds; measured JSON is included in exports/ |
| Logo and brand delivery | PASS | App mark, favicon and wordmark served as image/svg+xml; README logo rendered from the tracked SVG |

The browser suite verifies profile creation and recoverable validation errors, an absent-secret connection outcome, probe import and immutable revisions, audit freezing/cancellation/execution, evidence charts, persisted audit history, report export and evidence purging. It checks responsive widths and severe JavaScript errors. The audit-verdict unit tests cover randomized ranks, forged calibration bounds, exact configuration/budget binding and conservative verdicts. Controlled local HTTP fixtures exercise retry, timeout, malformed-response and authentication-error behavior.

## Fixes made for the final product

- Installed the real React/TypeScript/Vite toolchain and generated its complete dependency lock.
- Corrected six React form-event types and the Python 3.12/Pydantic import failure.
- Added a consistent original branching-M mark, favicon and SVG README wordmark.
- Made documented .env settings load without overriding existing process environment values; retained compatibility with legacy origin names.
- Served branding SVGs explicitly rather than returning the SPA HTML fallback.
- Corrected Windows interpreter-tree cleanup so the server and its file handles stop together.
- Made browser verification wait for the saved profile and its connection result instead of racing the asynchronous table refresh.

## Meaning and scope

ModelTrace 1.0.0 is a **local, single-user software product**. Initial setup downloads dependencies. Its synthetic demo needs no provider key or GPU after setup. Real endpoints need explicit profiles, secrets in backend environment variables, trusted references and bounded budgets. No paid provider calls were made during release verification.

Behavioral inconsistency is evidence about score distributions under the frozen tested conditions. It does not establish exact weights, quantization, parameter count or provider intent. Non-rejection does not prove model identity. A consistency verdict requires a matched calibration meeting the declared power adequacy policy for named alternatives.

The measured synthetic benchmark is software verification, not real hosted-LLM detection accuracy. Zero false rejections in 30 toy trials does not establish a 1% empirical false-positive rate. Wilson intervals have approximate nominal coverage; separate alternative intervals do not create a simultaneous confidence guarantee. The real HF test is a small local integration fixture, not an evaluation of every large checkpoint, precision or device combination.

Public multi-user hosting, paid-provider empirical benchmarks, every third-party checkpoint and native macOS execution are not certified by this release. These are deployment and evaluation boundaries, not an experimental preview label.

## Evidence and history

Current evidence: `release-evidence/github-final-verification.json`, the linked GitHub run, compiled assets, the release manifest and controlled local logs. The earlier sandbox-limited report is retained as `release-evidence/historical-local-review.md`; its BLOCKED/NOT RUN entries describe that past environment and are superseded by the current runner evidence.
