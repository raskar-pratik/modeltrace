"""Adapter B: Real OpenAI-compatible HTTP chat-completions adapter.

Features:
- Strict outbound URL & IP validation against operator allowlists (blocks metadata/private IPs unless explicitly allowlisted as a local inference origin; fails closed on DNS rebinding).
- Socket-level destination IP pinning via `create_pinned_httpx_transport` with TLS hostname verification (`verify=True`, SNI `server_hostname` preserved) and deliberate proxy disabling (`trust_env=False`).
- Resolves API keys solely from validated MODELTRACE_PROVIDER_* backend environment variables; scrubs secrets from all errors and metadata.
- Disables HTTP redirects (`follow_redirects=False`) and caps response payload size.
- Implements bounded retries with backoff for transient 429/5xx/timeout errors while recording exact retry counts and classifying 401/403, 429, 5xx, timeout, network, and malformed-JSON errors.
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from modeltrace.adapters.base import AdapterError, GenerationResult
from modeltrace.config import get_env_settings
from modeltrace.utils.security import (
    SecurityViolationError,
    canonicalize_origin,
    create_pinned_httpx_transport,
    redact_text,
    resolve_secret_value,
    validate_outbound_url,
)


class OpenAICompatibleAdapter:
    """Real HTTP chat-completions adapter for OpenAI-compatible reference and target APIs."""

    def __init__(
        self,
        profile: Dict[str, Any],
        allowed_outbound_origins: List[str],
        allowed_local_inference_origins: List[str],
        max_response_bytes: Optional[int] = None,
        resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
    ) -> None:
        self.profile = profile
        self.allowed_outbound_origins = allowed_outbound_origins
        self.allowed_local_inference_origins = allowed_local_inference_origins
        env = get_env_settings()
        self.max_response_bytes = max_response_bytes or env.max_response_bytes
        self.timeout_seconds = float(profile.get("timeout_seconds", 15.0))
        self.max_retries = int(profile.get("max_retries", 2))
        self.resolver_fn = resolver_fn
        self.last_connected_peer_ips: List[str] = []

    def _build_endpoint_url(self) -> str:
        base_url = (self.profile.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise AdapterError(
                "MISSING_BASE_URL",
                "Profile base_url is required for openai_compatible adapter.",
            )
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def check_capability(self, execute_network_probe: bool = False) -> Dict[str, Any]:
        """Validate profile configuration, secret env presence, and outbound allowlist compliance."""
        try:
            endpoint_url = self._build_endpoint_url()
            validated_url, origin, resolved_ips = validate_outbound_url(
                endpoint_url,
                self.allowed_outbound_origins,
                self.allowed_local_inference_origins,
                resolve_dns=True,
                resolver_fn=self.resolver_fn,
            )
            env_var = self.profile.get("api_key_env_var")
            if env_var:
                _ = resolve_secret_value(env_var)
        except (SecurityViolationError, AdapterError) as exc:
            code = getattr(exc, "code", getattr(exc, "error_code", "CAPABILITY_CHECK_FAILED"))
            return {
                "status": "unavailable",
                "adapter_type": "openai_compatible",
                "is_demo": False,
                "reason_code": code,
                "detail": redact_text(str(exc)),
                "supports_generation": False,
                "supports_local_logit_scoring": False,
            }

        result: Dict[str, Any] = {
            "status": "available",
            "adapter_type": "openai_compatible",
            "is_demo": False,
            "endpoint_origin": origin,
            "resolved_ips": resolved_ips,
            "secret_env_configured": bool(self.profile.get("api_key_env_var")),
            "supports_generation": True,
            "supports_local_logit_scoring": False,
            "declared_identity_trusted": False,
            "socket_ip_pinning": True,
            "trust_env_proxies": False,
            "note": "Declared model names and response headers are recorded as untrusted provider claims.",
        }

        if execute_network_probe:
            gen = self.generate(
                prompt_id="conn_test",
                prompt_text="Reply with the single word OK.",
                temperature=0.0,
                top_p=1.0,
                max_tokens=8,
            )
            result["connection_test"] = {
                "http_status": gen.http_status,
                "latency_ms": round(gen.latency_ms, 2),
                "declared_model": gen.declared_model,
                "finish_reason": gen.finish_reason,
                "sample_preview": gen.response_text[:80],
            }
        return result

    def generate(
        self,
        prompt_id: str,
        prompt_text: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int] = None,
        on_attempt_hook: Optional[Callable[[int], None]] = None,
    ) -> GenerationResult:
        """Execute a bounded chat-completions request with socket-pinned IP transport and classified error handling."""
        endpoint_url = self._build_endpoint_url()
        try:
            validated_url, origin, resolved_ips = validate_outbound_url(
                endpoint_url,
                self.allowed_outbound_origins,
                self.allowed_local_inference_origins,
                resolve_dns=True,
                resolver_fn=self.resolver_fn,
            )
            norm_local = {
                canonicalize_origin(o)
                for o in self.allowed_local_inference_origins
                if o.strip()
            }
            is_local_allowed = origin in norm_local

            secret_val: Optional[str] = None
            env_var = self.profile.get("api_key_env_var")
            if env_var:
                secret_val = resolve_secret_value(env_var)
        except SecurityViolationError as exc:
            raise AdapterError(
                error_code=exc.code,
                message=redact_text(exc.message),
                http_status=None,
                retryable=False,
            ) from exc

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ModelTrace/0.1.0",
        }
        if secret_val:
            headers["Authorization"] = f"Bearer {secret_val}"

        messages: List[Dict[str, str]] = []
        extra = self.profile.get("extra_config") or {}
        if extra.get("system_prompt"):
            messages.append({"role": "system", "content": str(extra["system_prompt"])})
        messages.append({"role": "user", "content": prompt_text})

        payload: Dict[str, Any] = {
            "model": self.profile.get("model_identifier"),
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": min(int(max_tokens), 128),
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        last_err: Optional[AdapterError] = None
        for retry_idx in range(self.max_retries + 1):
            if on_attempt_hook is not None and retry_idx > 0:
                on_attempt_hook(retry_idx)

            t0 = time.perf_counter()
            try:
                transport, pinned_backend = create_pinned_httpx_transport(
                    url_str=validated_url,
                    pinned_ips=resolved_ips,
                    allow_local_inference=is_local_allowed,
                    verify_tls=True,
                    resolver_fn=self.resolver_fn,
                )
                timeout_cfg = httpx.Timeout(
                    self.timeout_seconds, connect=min(5.0, self.timeout_seconds)
                )
                with httpx.Client(
                    transport=transport,
                    timeout=timeout_cfg,
                    follow_redirects=False,
                    trust_env=False,
                    verify=True,
                ) as client:
                    with client.stream("POST", validated_url, headers=headers, json=payload) as resp:
                        self.last_connected_peer_ips = list(pinned_backend.connected_peer_ips)
                        status_code = resp.status_code
                        if 300 <= status_code < 400:
                            raise AdapterError(
                                "REDIRECT_FORBIDDEN",
                                f"Endpoint returned redirect status {status_code}; redirects are forbidden by security policy.",
                                http_status=status_code,
                                retryable=False,
                                retry_count=retry_idx,
                            )

                        chunks: List[bytes] = []
                        total_bytes = 0
                        for chunk in resp.iter_bytes():
                            total_bytes += len(chunk)
                            if total_bytes > self.max_response_bytes:
                                raise AdapterError(
                                    "RESPONSE_TOO_LARGE",
                                    f"Response exceeded max_response_bytes ({self.max_response_bytes}).",
                                    http_status=status_code,
                                    retryable=False,
                                    retry_count=retry_idx,
                                )
                            chunks.append(chunk)
                        raw_body = b"".join(chunks).decode("utf-8", errors="replace")

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if status_code in (401, 403):
                    raise AdapterError(
                        "AUTH_ERROR",
                        f"Authentication failed with HTTP {status_code}: {redact_text(raw_body[:200], [secret_val] if secret_val else None)}",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                if status_code == 429:
                    raise AdapterError(
                        "RATE_LIMIT_429",
                        f"Rate limited (HTTP 429) by endpoint: {redact_text(raw_body[:200], [secret_val] if secret_val else None)}",
                        http_status=429,
                        retryable=True,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                if status_code >= 500:
                    raise AdapterError(
                        "SERVER_ERROR",
                        f"Upstream server error HTTP {status_code}: {redact_text(raw_body[:200], [secret_val] if secret_val else None)}",
                        http_status=status_code,
                        retryable=True,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                if status_code >= 400:
                    raise AdapterError(
                        "HTTP_CLIENT_ERROR",
                        f"Request rejected with HTTP {status_code}: {redact_text(raw_body[:200], [secret_val] if secret_val else None)}",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                try:
                    data = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    raise AdapterError(
                        "MALFORMED_RESPONSE",
                        f"Endpoint returned non-JSON body: {redact_text(raw_body[:160], [secret_val] if secret_val else None)}",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    ) from exc

                if (
                    not isinstance(data, dict)
                    or "choices" not in data
                    or not isinstance(data["choices"], list)
                    or len(data["choices"]) == 0
                ):
                    raise AdapterError(
                        "MALFORMED_RESPONSE",
                        "Response JSON missing non-empty 'choices' list.",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                first_choice = data["choices"][0]
                if not isinstance(first_choice, dict):
                    raise AdapterError(
                        "MALFORMED_RESPONSE",
                        "First entry in 'choices' is not an object.",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                msg = first_choice.get("message")
                if (
                    not isinstance(msg, dict)
                    or "content" not in msg
                    or not isinstance(msg["content"], str)
                ):
                    raise AdapterError(
                        "MALFORMED_RESPONSE",
                        "Choice missing valid 'message.content' string.",
                        http_status=status_code,
                        retryable=False,
                        retry_count=retry_idx,
                        latency_ms=elapsed_ms,
                    )

                content_text = msg["content"]
                finish_reason = str(first_choice.get("finish_reason") or "stop")
                declared_model = str(
                    data.get("model") or self.profile.get("model_identifier") or "unknown"
                )
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                completion_tokens = int(
                    usage.get("completion_tokens") or len(content_text.split())
                )

                return GenerationResult(
                    response_text=content_text,
                    token_count=max(1, completion_tokens) if content_text else 0,
                    finish_reason=finish_reason,
                    declared_model=declared_model,
                    latency_ms=elapsed_ms,
                    http_status=status_code,
                    retry_count=retry_idx,
                    observable_meta={
                        "declared_model": declared_model,
                        "declared_model_untrusted": True,
                        "system_fingerprint": data.get("system_fingerprint"),
                        "usage": usage,
                        "requested_model": self.profile.get("model_identifier"),
                        "requested_temperature": temperature,
                        "requested_top_p": top_p,
                        "requested_max_tokens": min(int(max_tokens), 128),
                        "pinned_peer_ips": list(self.last_connected_peer_ips),
                    },
                )

            except SecurityViolationError as exc:
                raise AdapterError(
                    error_code=exc.code,
                    message=redact_text(exc.message, [secret_val] if secret_val else None),
                    http_status=None,
                    retryable=False,
                    retry_count=retry_idx,
                ) from exc
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                last_err = AdapterError(
                    "TIMEOUT",
                    f"Request timed out after {self.timeout_seconds}s: {redact_text(str(exc), [secret_val] if secret_val else None)}",
                    http_status=None,
                    retryable=True,
                    retry_count=retry_idx,
                    latency_ms=elapsed_ms,
                )
            except httpx.RequestError as exc:
                # Unwrap SecurityViolationError if raised inside httpcore/PinnedIPSyncBackend
                cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
                if isinstance(cause, SecurityViolationError):
                    raise AdapterError(
                        error_code=cause.code,
                        message=redact_text(cause.message, [secret_val] if secret_val else None),
                        http_status=None,
                        retryable=False,
                        retry_count=retry_idx,
                    ) from cause
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                last_err = AdapterError(
                    "NETWORK_ERROR",
                    f"Network communication error: {redact_text(str(exc), [secret_val] if secret_val else None)}",
                    http_status=None,
                    retryable=True,
                    retry_count=retry_idx,
                    latency_ms=elapsed_ms,
                )
            except AdapterError as exc:
                last_err = exc

            if last_err is not None:
                if not last_err.retryable or retry_idx >= self.max_retries:
                    raise last_err
                backoff = min(0.25 * (2**retry_idx), 1.0)
                time.sleep(backoff)

        assert last_err is not None
        raise last_err
