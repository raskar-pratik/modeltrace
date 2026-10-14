# ModelTrace V1 — Security & Threat Model (`SECURITY.md`)

## 1. Deployment Scope & Explicit Non-Claims

- **Localhost Single-User Workbench:** ModelTrace binds to `127.0.0.1:8000` by default (`MODELTRACE_HOST=127.0.0.1`). It is architected for a single operator running on a local workstation or private VM.
- **No Public Multi-User Hosting Claims:** ModelTrace V1 intentionally omits multi-tenant authentication, RBAC, billing, and public internet exposure. Do **not** bind ModelTrace to `0.0.0.0` on an untrusted network.

## 2. API Key Isolation, Immutable Manifest Snapshots & Secret Redaction

1. **Environment-Variable-Only Key References (`^MODELTRACE_PROVIDER_[A-Z0-9_]+$`):**
   - Endpoint profiles store only the *name* of an environment variable (e.g., `MODELTRACE_PROVIDER_TOGETHER_KEY`).
   - `validate_secret_env_var_name()` in `modeltrace/utils/security.py` rejects raw API keys (`sk-...`, `Bearer ...`), lowercase strings, and non-`MODELTRACE_PROVIDER_*` environment variables (`PATH`, `HOME`, `OPENAI_API_KEY`).
   - When an audit is created, `freeze_profile_snapshot()` records the immutable profile configuration and `api_key_env_var` reference name in `manifest_json`—never the resolved secret key value.
2. **Defense-in-Depth Secret Redaction (`redact_text`, `redact_secrets`):**
   - Before any audit event, adapter error, capability diagnostic, or report export (`JSON`, `CSV`, `HTML`) is written to SQLite or returned over HTTP, `redact_secrets()` recursively scrubs all current `MODELTRACE_PROVIDER_*` values, `Bearer` tokens, `sk-*` keys, and sensitive dictionary keys.

## 3. Outbound Network, Socket-Level IP Pinning & TLS Verification

All outbound requests in `OpenAICompatibleAdapter` enforce both pre-flight origin/IP validation and **socket-level IP pinning**:
1. **Strict Scheme & Credential Check:** Only `http://` and `https://` URLs are permitted; embedded URL credentials (`http://user:pass@host`) are rejected (`CREDENTIALS_IN_URL_FORBIDDEN`). Public outbound origins must use `https://` (`HTTPS_REQUIRED_FOR_PUBLIC_ORIGIN`).
2. **Two-Tier Origin Allowlisting:**
   - **Public Outbound Allowlist (`allowed_outbound_origins`):** Requires `https://`.
   - **Local Inference Allowlist (`allowed_local_inference_origins`):** Loopback/private origins (`http://127.0.0.1:8080`, `http://127.0.0.1:11434`) are blocked (`PRIVATE_DESTINATION_FORBIDDEN`) unless the exact `scheme://host:port` origin is in `allowed_local_inference_origins`.
3. **DNS Resolution & Unconditional Cloud Metadata Blocking (`169.254.169.254`):**
   - `resolve_and_validate_ips()` resolves A/AAAA records and blocks link-local and cloud metadata ranges (`169.254.0.0/16`, `169.254.169.254`, `fe80::/10`, `100.100.100.200/32`) unconditionally (`METADATA_DESTINATION_FORBIDDEN`).
4. **Socket-Level Destination IP Pinning (`PinnedIPSyncBackend`, `create_pinned_httpx_transport`):**
   - Because validating DNS before `httpx.Client` alone leaves a TOCTOU DNS rebinding window, `OpenAICompatibleAdapter` uses a custom `httpcore.backends.sync.SyncBackend` subclass (`PinnedIPSyncBackend`).
   - `PinnedIPSyncBackend.connect_tcp()` verifies that the requested `(host, port)` matches the approved endpoint (`UNEXPECTED_CONNECT_TARGET`), re-verifies DNS resolution against the pre-flight IP set (`DNS_REBINDING_DETECTED`), opens the TCP socket directly to `(pinned_ip, expected_port)`, and verifies `sock.getpeername()[0]` against `pinned_ips` and `validate_peer_ip()`.
   - When TLS is started (`SyncStream.start_tls`), `httpcore` passes `server_hostname=expected_host` with `ssl.create_default_context()` (`verify=True`), preserving full TLS certificate hostname verification and SNI while the underlying TCP socket is pinned to the validated IP.
5. **Deliberate Proxy Disabling (`trust_env=False`) & Redirect Blocking (`follow_redirects=False`):**
   - `httpx.Client` and `httpx.HTTPTransport` are initialized with `trust_env=False` (so ambient `HTTP_PROXY`/`HTTPS_PROXY` variables cannot reroute requests or leak API keys) and `follow_redirects=False` (any `3xx` redirect is aborted with `REDIRECT_FORBIDDEN`).

## 4. Inbound Localhost `Host` & `Origin` Validation

FastAPI middleware (`security_host_origin_and_json_guard` in `modeltrace/api/app.py`) calls `validate_inbound_localhost_request()` on every inbound HTTP request:
1. **Explicit Localhost `Host` Allowlist (`DISALLOWED_HOST_HEADER`):**
   - The inbound `Host` header is parsed and its hostname is checked against `DEFAULT_ALLOWED_LOCALHOST_HOSTS` (`127.0.0.1`, `localhost`, `::1`, `[::1]`, `testserver`, `modeltrace.local`).
   - Any request carrying an unapproved `Host` header (e.g., `Host: evil.example.com` or a DNS-rebound domain `Host: rebind.attacker.invalid:8000`) is rejected with `400 Bad Request` (`DISALLOWED_HOST_HEADER`).
2. **Strict `Origin` & `Referer` Validation (`DISALLOWED_ORIGIN`):**
   - ModelTrace **never** trusts the inbound `Host` header to define a valid same-origin `Origin`. Even if an attacker sends matching `Host: evil.example.com` and `Origin: http://evil.example.com`, the request is rejected.
   - `Origin` (and `Referer` on state-mutating requests) must match `env.allowed_frontend_origins` or an explicit `http://<allowed_localhost_host>:<port>` origin.
3. **Upload Size Cap & JSON Content-Type Guard:**
   - State-changing requests exceeding `MODELTRACE_MAX_UPLOAD_BYTES` (2 MB default) are rejected with `413 Payload Too Large` (`UPLOAD_TOO_LARGE`); non-JSON bodies are rejected with `415 Unsupported Media Type`.

## 5. Stored Response Rendering, HTML Escaping & CSV Formula Neutralization

- **Browser & Report XSS Prevention:** Untrusted model completions are rendered via React JSX text nodes in the UI and escaped via `html.escape(..., quote=True)` in exported HTML reports.
- **CSV Formula Injection Neutralization (`neutralize_csv_cell`):** Any string cell exported to CSV whose leading non-space character is `=`, `+`, `-`, `@`, `\t`, or `\r` is prefixed with `'` so spreadsheet applications never execute embedded formulas.
