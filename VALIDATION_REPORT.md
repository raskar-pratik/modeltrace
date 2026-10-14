# ModelTrace 1.0 — independent release verification

Verified on 14 October 2026. Release scope: **versioned single-user local auditing product**. Public hosting was not configured or published.

## GitHub release verification update

The former preview label reflected incomplete verification in the local sandbox. GitHub CI has since completed successfully for commit `31a4712f52d4e4182d30dfa9f099d1b3940c986d`:

- [Successful release workflow](https://github.com/raskar-pratik/modeltrace/actions/runs/38580108372), completed 2026-10-14.
- Linux real React/TypeScript/Vite build and connected browser workflow: PASS; 28 tests passed, 1 optional HuggingFace check skipped.
- Multi-stage Docker image, container health, static assets and bounded demo: PASS.

The native Windows and actual HuggingFace jobs were added with the branded product update. Their results are pending until that commit's workflow completes. The historical local results below retain the original environment limits; they no longer imply that browser or Docker execution is unverified everywhere.

## Changes completed

- Generated a real npm lock with 123 package entries, including esbuild, PostCSS, Rollup, React and React DOM. Setup, Docker and CI now require this lock with npm ci.
- Fixed six React form handlers that failed actual TypeScript checking. Built and packaged the real React 18.2.0 / Vite 6.4.4 UI. Removed the substitute React runtime and partial AST checker.
- Fixed the Python 3.12 import failure by pinning Pydantic 1.10.26 and its compatible typing dependency. Added missing transitive Python pins and aligned release metadata to 1.0.0. Removed tests from the installed production package list so the Docker runtime layout can build it. Supported Python range is explicitly 3.11–3.12; execution here used 3.12.
- Made the launcher reject missing or substitute UI assets and select the Windows virtualenv path. Retained fail-fast POSIX/PowerShell setup.
- Fixed browser test input edits for real React controlled forms, isolated local HTTP fixture traffic from ambient proxies, and added cleanup even if browser setup fails.
- Corrected CI browser binary discovery to use the action's declared outputs. Release CI requires a browser rather than silently skipping it. Docker verifies real assets; GitHub CI itself was not run in this session.
- Replaced outdated documentation and removed inherited screenshots of the previous substitute-runtime build. Added START_HERE.md and a no-provider-call local HTTP smoke script.

## Executed verification

Logs and exact commands are included in `release-evidence/`. Initial failures are retained in browser diagnostic logs; they are not represented as passes.

| Check | Result | Evidence |
|---|---|---|
| Online isolated Python dependency installation, editable package, pip check | PASS | Installed actual PyPI packages; no system-site-packages or dependency shims |
| Clean npm ci | PASS | Actual installed React, TypeScript and Vite; complete generated lock |
| Actual tsc --noEmit | PASS | Six form-event errors repaired |
| TypeScript negative control | PASS | Temporary string-to-number assignment failed with TS2322 and exit 2; restored source before build |
| Vite production build | PASS | Real JS/CSS assets and SPA HTML included |
| npm audit | PASS | Zero known advisories reported for this resolved frontend dependency tree at verification time; not a general security guarantee |
| Backend and setup regressions | PASS | 27 passed, 1 optional HF test skipped, 1 browser test deselected |
| Migrations and launcher prerequisite check | PASS | Idempotent migration path; launcher detects real assets |
| Full synthetic CLI audits and exports | PASS | Substitution: INCONSISTENT (p=0.0001); genuine with compatible synthetic calibration: CONSISTENT_WITHIN_TESTED_LIMITS (p=0.38); genuine without calibration: INCONCLUSIVE (p=0.38) |
| Live localhost HTTP delivery | PASS | Health, root HTML, /audits SPA route, JS, CSS and favicon served over HTTP |
| Separated-seed synthetic benchmark | PASS | Nine conditions, 30 evaluation trials each, 25 calibration trials per named alternative; newly generated JSON included |
| Fresh source copy and separate Python virtualenv | PASS | Fresh install/build/test/smoke logs included; runtime state and dependency directories excluded |
| ZIP integrity and payload manifest | PASS | Manifest checks every payload file; manifest excludes its own hash |

Backend checks cover rank/tie handling, Monte Carlo testing, forged calibration bounds, exact configuration binding, worker snapshots, local HTTP retry/error fixtures, migrations, redaction and transport-security regressions. Mocked sharded-checkpoint tests are software unit tests, not evidence of a real HuggingFace model run.

## Remaining verification limits

| Check | Status | Reason |
|---|---|---|
| Production browser workflow | BLOCKED | Provisioned Chrome/ChromeDriver, but Chrome cannot create a required Unix socket: Operation not permitted. The Selenium attempt failed before a browser session started. Responsive layout and browser interactions remain unverified for this real build. |
| Docker image build/run | BLOCKED | Docker and Podman are absent. The multi-stage recipe and CI job are included, not certified. |
| Native Windows/PowerShell execution | NOT RUN | PowerShell is absent. Guard structure is checked by a test; actual Windows behavior is not claimed. |
| Actual HuggingFace generation/scoring | NOT RUN | Optional torch/transformers are not installed; corresponding integration test is skipped. |
| Live commercial LLM auditing | NOT RUN | No paid provider calls or credentials used. Synthetic benchmarks are not real LLM accuracy. |
| GitHub Actions workflow | NOT RUN | Workflow prepared locally; no remote repository/run was selected. |

## Newly measured synthetic outcomes

All evaluation conditions use alpha=0.01 and 9,999 Monte Carlo null simulations. Calibration seeds are separated from evaluation seeds. Counts below were rerun from this source.

| Condition | Rejections | Empirical rate | Two-sided nominal 95% Wilson interval |
|---|---:|---:|---|
| `genuine_null_logit_n20_m8` | 0/30 | 0.000 | [0.0000, 0.1135] |
| `weak_alt_5pct_logit_n20_m8` | 0/30 | 0.000 | [0.0000, 0.1135] |
| `mixed_15pct_logit_n20_m8` | 3/30 | 0.100 | [0.0346, 0.2562] |
| `mixed_35pct_logit_n20_m8` | 15/30 | 0.500 | [0.3315, 0.6685] |
| `mixed_65pct_logit_n20_m8` | 30/30 | 1.000 | [0.8865, 1.0000] |
| `substituted_100pct_logit_n20_m8` | 30/30 | 1.000 | [0.8865, 1.0000] |
| `mixed_35pct_logit_low_budget_n10_m4` | 6/30 | 0.200 | [0.0951, 0.3731] |
| `substituted_100pct_length_only_n20_m8` | 30/30 | 1.000 | [0.8865, 1.0000] |
| `genuine_null_length_only_n20_m8` | 0/30 | 0.000 | [0.0000, 0.1135] |

Zero false alarms in 30 toy trials does not establish a 1% empirical false-positive rate: the Wilson upper bound is about 11.35%. Weak 5% routing changes were not detected at the tested screening budget. Calibration requires empirical power and the recomputed nominal 95% Wilson lower bound to both meet 0.80 for each named alternative. Wilson coverage is approximate and per-alternative; it does not establish simultaneous coverage or sensitivity to unseen alternatives.

## Start and remaining release checks

Follow START_HERE.md. Initial setup needs registry access; the demo needs no key or GPU once dependencies are installed. Open http://127.0.0.1:8000 after launch. Live provider profiles require explicit configuration and can use paid quota.

On a suitable runner, run the required browser workflow and Docker job before calling the packaging pipeline fully verified. Native Windows and actual checkpoint execution need their corresponding platforms/dependencies. Public multi-user hosting needs its own authentication and deployment design.

Build provenance, asset hashes, actual command results and software versions are in `release-evidence/build-provenance.json` and `release-evidence/commands.json`.

Primary dependency references used during repair: [Pydantic 1.10.26 publisher record](https://pypi.org/project/pydantic/1.10.26/) and [setup-chrome output documentation](https://github.com/browser-actions/setup-chrome#outputs). Verification claims above come from the included execution logs, not these documentation pages.
