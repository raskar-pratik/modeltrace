# ModelTrace — System Architecture

## 1. Architectural Overview

ModelTrace is designed as a **localhost single-user modular monolith** written in Python 3.11. It avoids distributed queue brokers, external caches, or multi-tenant cloud services in favor of a deterministic SQLite persistence layer and a single background worker thread coordinated through atomic database state transitions.

```
+-----------------------------------------------------------------------+
|                   Clients (CLI / Stage 2 React UI)                    |
+-----------------------------------+-----------------------------------+
                                    | HTTP JSON (127.0.0.1) / CLI Calls
                                    v
+-----------------------------------------------------------------------+
|                     FastAPI Application Layer                         |
|  - Origin & Content-Type Guards   - Secret Redaction Middleware       |
|  - Profiles / Probes / Calibrations / Audits / Settings / Exports     |
+-----------------+-----------------------------------+-----------------+
                  |                                   |
                  v                                   v
+-----------------------------------+   +-------------------------------+
|    Database-Backed Single Worker  |   |   Statistical & Export Engine |
|  - Poll / Wake Event Queue        |   |  - Frozen SeedSequence RNGs   |
|  - Startup Interruption Recovery  |   |  - Randomized Tie Uniform z_i |
|  - Pre-Call Attempt Persistence   |   |  - Cramér-von Mises MC Test   |
|  - Conservative Explicit Resume   |   |  - Holm & Bootstrap CI        |
+-----------------+-----------------+   +-------------------------------+
                  |
                  v
+-----------------------------------------------------------------------+
|                        Adapter Abstraction Layer                      |
|  [A] SyntheticDemoAdapter  [B] OpenAICompatibleAdapter  [C] LocalHF   |
|                            (Outbound URL/IP Guard)    (Teacher-Force) |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                  SQLite Persistence Layer (WAL Mode)                  |
|  schema_migrations | profiles | probe_revisions | calibrations        |
|  audits | sample_attempts | sample_responses | prompt_scores | events |
+-----------------------------------------------------------------------+
```

## 2. Module Responsibilities

- `modeltrace/config.py`: Central runtime configuration, environment variable parsing (`MODELTRACE_DB_PATH`, `MODELTRACE_LOCAL_MODELS_DIR`, `MODELTRACE_HOST`, `MODELTRACE_PORT`), and constants.
- `modeltrace/utils/security.py`:
  - Validates provider secret references (`^MODELTRACE_PROVIDER_[A-Z0-9_]+$`) and resolves secrets exclusively at adapter invocation time.
  - Redacts any resolved secrets, Bearer tokens, or API key patterns from strings, dictionaries, logs, and error tracebacks.
  - Enforces exact-origin (`scheme://host:port`) outbound allowlists, rejects credentials in URLs, disables redirects, caps response byte lengths, resolves DNS, and blocks private/loopback/link-local/metadata IP addresses unless the exact origin is explicitly registered in `allowed_local_inference_origins` (with `169.254.169.254` unconditionally denied).
- `modeltrace/db/`:
  - `migrations.py`: Deterministic, idempotent SQL migration runner tracking applied versions in `schema_migrations`.
  - `base.py` & `repository.py`: Dual-compatible database layer supporting SQLAlchemy Engine/Session when installed and standard-library `sqlite3` with WAL mode, foreign keys, and atomic transactions so all persistence features work identically in offline environments.
- `modeltrace/adapters/`:
  - `base.py`: Abstract adapter contract (`generate`, `score_reference_logits`, `test_connection`, `check_capability`).
  - `synthetic_demo.py`: Seeded stochastic reference/target generator and scorer supporting `genuine`, `substituted`, and `mixed_routing` scenarios, explicitly tagged `is_demo=True`.
  - `openai_compatible.py`: Real HTTP `/v1/chat/completions` client using `httpx` with outbound security validation, bounded retries, exponential backoff on 429/5xx, strict timeouts, and malformed response detection.
  - `local_transformers.py`: Local HuggingFace Transformers reference generator and full-vocabulary teacher-forcing logit-rank scorer (`trust_remote_code=False`, `local_files_only=True`), plus a deterministic causal LM logit-scoring implementation and tiny random-weight fixture helper.
- `modeltrace/services/`:
  - `probes.py`: Built-in 24-prompt suite across 5 task families (`extraction`, `reasoning`, `code`, `multilingual`, `formatting`), JSON/JSONL import validator, and immutable revision creator.
  - `stats.py`: Pure, deterministic statistical engine implementing exact tie-broken uniform transformation $z_i = (a_i + J_i + U_i)/(m+1)$, Cramér–von Mises test statistic $W^2$, Monte Carlo uniform null simulation ($B \ge 9999$), Holm step-down multiplicity correction, prompt/family bootstrap confidence intervals, and calibration compatibility evaluation.
  - `worker.py`: Single background worker thread and job coordinator. Handles startup recovery (marking unfinished `RUNNING` jobs as `INTERRUPTED`), pre-call attempt creation (`PLANNED -> IN_FLIGHT`), budget cap enforcement, cooperative cancellation, and explicit resume.
  - `exporter.py`: Generates reproducible JSON, CSV, and self-contained HTML reports (styled with the warm monochrome editorial rules of `minimalist-ui`) and records SHA-256 integrity hashes.
- `modeltrace/api/app.py`: FastAPI application with localhost origin/CORS checks, JSON content-type enforcement on state-changing routes, and complete REST endpoints.
- `modeltrace/cli.py`: Command-line interface for database migrations, capability checks, demo and live audits, resume, exports, and server launch.

## 3. Job Lifecycle & Crash Recovery Semantics

1. **Creation & Preview:** `POST /api/audits/preview` computes exact attempt requirements without database writes. `POST /api/audits` freezes the audit manifest (probe revision hash, sampling parameters, $\alpha$, derived RNG seeds, attempt/duration caps) in state `CREATED` (or immediately transitions to `RUNNING` if `start_immediately=True` is explicitly requested).
2. **Single-Start Guard:** `POST /api/audits/{id}/start` performs an atomic conditional `UPDATE audits SET execution_state = 'RUNNING' WHERE id = ? AND execution_state = 'CREATED'`. Concurrent start calls fail with `409 Conflict`.
3. **Pre-Call Attempt Persistence:** Before calling any target or reference adapter, the worker inserts a `sample_attempts` row in status `IN_FLIGHT` and commits. Every retry increments the attempt counter and persists a new attempt record counted against `max_target_attempts`, `max_reference_attempts`, and `max_total_attempts`.
4. **Startup Interruption:** When the application or worker initializes, any audit left in `RUNNING` or `STARTING` state is atomically marked `INTERRUPTED` with verdict `INCONCLUSIVE`, and any `IN_FLIGHT` attempt is marked `UNCERTAIN_PRIOR_ATTEMPT`. Paid external work is never silently repeated.
5. **Conservative Explicit Resume:** When the operator invokes `POST /api/audits/{id}/resume`, the worker reuses all `COMPLETED` prompt responses, keeps `UNCERTAIN_PRIOR_ATTEMPT` rows counted against the total attempt budget (recording in the event log that exactly-once external delivery cannot be guaranteed after an unclean crash), and executes only the remaining missing prompt attempts if sufficient budget remains.
