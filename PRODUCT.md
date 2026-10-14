# ModelTrace — Product Specification

> **Know what your API is serving.**

## 1. Mission & Scope

ModelTrace is a localhost, single-user auditing workbench and CLI that tests whether an open-weight model API behaves consistently with a trusted reference configuration under black-box or gray-box query access.

Commercial and self-hosted inference endpoints frequently claim to serve a specific open-weight checkpoint (e.g., Llama-3, Qwen, Mistral) at a declared precision. In practice, endpoints may silently route traffic to quantized variants, distilled substitutes, altered chat templates, injected system prompts, or mixed model pools. ModelTrace gives engineering and evaluation teams a reproducible statistical harness to detect behavioral divergence under controlled prompts and sampling configurations.

## 2. Explicit Epistemic Boundaries (What ModelTrace Does and Does Not Prove)

ModelTrace evaluates **observable input-output behavior under a frozen test configuration**. Every API response, CLI output, and exported report enforces the following boundaries:

1. **No Proof of Weights or Intent:** Statistical consistency ($p > \alpha$) or inconsistency ($p \le \alpha$) concerns output distributions under the tested prompt suite and decoding settings. It is **not** cryptographic proof of model weights, parameter count, floating-point precision, provider honesty, or fraud.
2. **Confounders Can Cause Divergence:** Differences in chat template formatting, hidden system prompts, safety moderation layers, stop sequences, speculative decoding, tensor-parallel reductions, or hardware floating-point non-determinism can produce statistically significant divergence (`INCONSISTENT`) even when the underlying base weights are identical.
3. **Verdict Separation from Execution State:**
   - **Execution State** (`CREATED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`) describes whether the job finished collecting its required sample budget.
   - **Verdict** (`INCONSISTENT`, `INCONCLUSIVE`, `CONSISTENT_WITHIN_TESTED_LIMITS`) describes the statistical conclusion:
     - `INCONSISTENT`: Valid, complete sample rejects the null hypothesis of exchangeability at level $\alpha$.
     - `INCONCLUSIVE`: Sample is incomplete/invalid, OR the null hypothesis is not rejected ($p > \alpha$) and no compatible, independently validated power calibration artifact is attached.
     - `CONSISTENT_WITHIN_TESTED_LIMITS`: Valid, complete sample does not reject the null ($p > \alpha$) **and** a compatible calibration artifact demonstrates adequate sensitivity ($\ge 80\%$ empirical power across $\ge 20$ trials) against named alternative configurations at the tested sample budget and $\alpha$.
4. **Score Mode Scoping:**
   - **Primary Mode (`LOCAL_LOGIT_RANK`):** Uses a local Transformers reference checkpoint via teacher-forcing to compute the mean log vocabulary rank of target and reference response tokens under full-vocabulary reference logits.
   - **Fallback Mode (`LENGTH_DISTRIBUTION_ONLY`):** When local logit scoring is unavailable, evaluates a frozen Unicode response-length feature through the same rank engine. Reports and API payloads explicitly scope this mode as `LENGTH DISTRIBUTION ONLY`—it cannot verify token-level identity or inherit local-scoring sensitivity.
   - **Descriptive Metrics (`DESCRIPTIVE_ONLY`):** Normalized log-likelihood, latency, token counts, schema compliance, and response hashes are reported as descriptive diagnostics unless separately registered and Holm-corrected.

## 3. Core Capabilities (Stage 1)

- **Profile Management & Capability Checks:** Configure `synthetic_demo`, `openai_compatible`, and `local_transformers` profiles. Secrets are referenced strictly via `MODELTRACE_PROVIDER_*` backend environment variables. Connection tests verify outbound allowlist rules and record observable endpoint behavior without trusting declared model headers.
- **Immutable Probe Revisions:** Built-in original 24-prompt screening suite spanning extraction, reasoning, code generation, multilingual translation, and structured formatting (explicitly labeled unvalidated for universal identity discrimination). All edits or JSON/JSONL imports create new immutable revisions with SHA-256 content hashes.
- **Calibration Provenance:** Import or generate calibration artifacts that record empirical detection sensitivity against named alternatives at specific $(n, m, \alpha)$ budgets, enforcing compatibility checks before granting `CONSISTENT_WITHIN_TESTED_LIMITS`.
- **Deliberate, Budget-Capped Audit Execution:** Preview attempt budgets before execution. Persist every sample attempt prior to network invocation. Enforce hard caps on reference attempts, target attempts, total attempts (including retries), and duration. Support cooperative cancellation, crash interruption detection on restart, and conservative explicit resume.
- **Reproducible Exports:** Export audit manifests, prompt-level rank statistics, event logs, and self-contained HTML/JSON/CSV reports with SHA-256 integrity hashes and full secret redaction.
