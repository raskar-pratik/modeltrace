# ModelTrace 1.0 — experimental localhost preview

**Know what your API is serving.** ModelTrace compares an LLM API's observable behavior with a trusted reference under a frozen audit configuration.

This release includes the Python source, real React/TypeScript/Vite production assets, an npm-generated dependency lock, tests, CLI, and verification logs. See `VALIDATION_REPORT.md` for executed checks and remaining verification limits.

## Start on Linux or macOS

Install Python 3.11 or 3.12 and Node.js 20 or newer. First setup downloads dependencies; synthetic audits require no API key or GPU once installed.

```bash
bash scripts/setup.sh
.venv/bin/python3 scripts/launch.py
```

Open **http://127.0.0.1:8000**. Use New Audit → a quick synthetic demo. Every synthetic result is marked DEMO.

## Start on Windows

Install Python 3.11 or 3.12 and Node.js 20 or newer. In PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe .\scripts\launch.py
```

Windows setup has native-process exit-code guards. Actual Windows execution was not available during this release's verification.

## CLI demo

```bash
.venv/bin/modeltrace demo-audit --scenario substituted --n-prompts 20 --m-reference 8 --seed 424242
.venv/bin/modeltrace demo-audit --scenario genuine --with-calibration --n-prompts 20 --m-reference 8 --seed 424242
.venv/bin/modeltrace demo-audit --scenario genuine --no-calibration --n-prompts 20 --m-reference 8 --seed 424242
```

Run the commands in order using their default database, or supply `--db-path` for an isolated database. They exercise rejection, non-rejection with synthetic calibration, and non-rejection without calibration respectively.

## What the result means

- **INCONSISTENT:** the score distributions differ under the tested conditions.
- **INCONCLUSIVE:** evidence, calibration, or execution is insufficient.
- **CONSISTENT_WITHIN_TESTED_LIMITS:** non-rejection with a configuration-matched power calibration that meets the declared adequacy policy for its named alternatives.

The randomized rank transformation and Cramér–von Mises Monte Carlo test depend on exchangeability and independence assumptions. Calibration requires at least 20 trials per alternative, empirical power ≥0.80, and a recomputed two-sided nominal 95% Wilson lower bound ≥0.80. Wilson coverage is approximate; this is not a simultaneous guarantee across alternatives or a guarantee for unseen substitutions.

Synthetic results validate software behavior under the toy distributions only. They are not measured detection accuracy on hosted LLMs. Behavioral divergence does not reveal exact weights, quantization, parameter count, intent, or fraud. Non-rejection does not prove model identity.

## Real endpoints and local checkpoints

Create endpoint profiles in the UI. Store provider secrets only in backend environment variables named `MODELTRACE_PROVIDER_*`; profiles hold the variable name. Explicitly configure allowed outbound origins. A connection test or live audit can use the selected provider's quota. The bundled demo and smoke checks make no paid provider calls.

For a local checkpoint, install the optional adapter dependencies:

```bash
.venv/bin/python3 -m pip install -e '.[local-inference]'
```

Use a trusted pinned checkpoint and tokenizer with a valid native chat template or a validated override. Actual HuggingFace execution was not verified in this release environment; mocked sharded-adapter tests are distinct from it. See `SECURITY.md` and `STATISTICAL_METHOD.md` for the detailed contracts.

## Verify and rebuild

```bash
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run build
.venv/bin/python3 -m pip check
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/smoke_local.py
```

For mandatory browser verification, provision matching Chrome and ChromeDriver, export `MODELTRACE_CHROME_BINARY` and `MODELTRACE_CHROMEDRIVER_PATH`, and set `MODELTRACE_REQUIRE_BROWSER=1` before pytest. CI provisions these binaries and treats their absence as a failure. It also runs a TypeScript negative control and builds/smokes the Docker image. The workflow is included but has not been executed on GitHub from this session.

## Docker

```bash
cp .env.example .env
mkdir -p data exports
docker compose up --build
```

The host port binds only to 127.0.0.1. If the container cannot write the mounted directories, grant the container user write access to those directories. Docker build/run was not available in the verification environment.

This release is a **single-user localhost preview**. A public multi-user service needs a separate hosting/authentication design and release verification.
