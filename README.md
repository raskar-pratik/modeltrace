<p align="center">
  <img src="docs/assets/modeltrace-logo.svg" width="760" alt="ModelTrace — Know what your API is serving." />
</p>

<p align="center">
  <a href="https://github.com/raskar-pratik/modeltrace/actions/workflows/ci.yml"><img src="https://github.com/raskar-pratik/modeltrace/actions/workflows/ci.yml/badge.svg" alt="Release CI" /></a>
  <img src="https://img.shields.io/badge/version-1.0.0-232522" alt="Version 1.0.0" />
  <img src="https://img.shields.io/badge/runs-locally-CAB38A" alt="Runs locally" />
</p>

# ModelTrace

### Know what your API is serving.

**Version 1.0.0 · A local LLM auditing product**

ModelTrace compares an LLM API's observable behavior with a trusted reference under a frozen audit configuration.

Audit a target endpoint, inspect its evidence, and export a report from one workbench. ModelTrace includes:

| Capability | What you get |
|---|---|
| Endpoint profiles | OpenAI-compatible targets and trusted API or local references |
| Reproducible audits | Frozen prompts, profiles, seeds, sampling settings, and request budgets |
| Statistical evaluation | Randomized ranks, Monte Carlo CvM testing, and matched power calibration |
| Evidence explorer | Stored responses, rank distributions, per-prompt shifts, and audit history |
| Reports and CLI | JSON, CSV, and HTML exports; controlled demos without provider keys |
| Local deployment | React workbench and Python API on one origin; Windows/POSIX setup and Docker |

The release pipeline verifies the real TypeScript/Vite build, the connected browser workflow, Docker health and assets, and deterministic software tests. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for dated evidence and the precise verification scope.

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

The PowerShell setup checks every native-process exit code and stops on a failed install, typecheck, build, or migration. The release pipeline exercises native Windows setup and HTTP delivery.

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

Use a trusted pinned checkpoint and tokenizer with a valid native chat template or a validated override. The release pipeline exercises actual HuggingFace generation and teacher-forced scoring against a small locally constructed checkpoint. Large third-party checkpoints still require their own compatibility checks. See `SECURITY.md` and `STATISTICAL_METHOD.md` for the detailed contracts.

## Verify and rebuild

```bash
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run build
.venv/bin/python3 -m pip check
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/smoke_local.py
```

For mandatory browser verification, provision matching Chrome and ChromeDriver, export `MODELTRACE_CHROME_BINARY` and `MODELTRACE_CHROMEDRIVER_PATH`, and set `MODELTRACE_REQUIRE_BROWSER=1` before pytest. CI provisions these binaries and treats their absence as a failure. It also runs a TypeScript negative control and builds/smokes the Docker image. Native Windows setup and actual local HuggingFace execution have dedicated jobs in the same workflow.

## Docker

```bash
cp .env.example .env
mkdir -p data exports
docker compose up --build
```

The host port binds only to 127.0.0.1. If the container cannot write the mounted directories, grant the container user write access to those directories. The Docker image build, health endpoint, production assets, and bounded synthetic audit are exercised in release CI.

ModelTrace 1.0 is a **single-user local product**. It runs on your machine and keeps provider secrets in the backend. Internet-facing, multi-user hosting is outside this product's deployment scope.

## Logo

The branching **M** connects a reference trajectory with an observed trace. The charcoal, warm ivory, and muted brass palette is shared across the app mark, favicon, and README. Scalable assets are in [frontend/public](frontend/public) and [docs/assets](docs/assets).
