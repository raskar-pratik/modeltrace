"""Security utilities: secret env resolution, log/export redaction, outbound URL/IP guards, socket-level IP pinning, and inbound localhost Host/Origin validation."""

import ipaddress
import os
import re
import socket
import ssl
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import httpcore
import httpx
from httpcore.backends.sync import SyncBackend, SyncStream

SECRET_ENV_NAME_RE = re.compile(r"^MODELTRACE_PROVIDER_[A-Z0-9_]+$")
BEARER_TOKEN_RE = re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-+/=]{6,}")
SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")
KEY_ASSIGN_RE = re.compile(
    r"(?i)((?:api[_-]?key|secret|password|authorization|token)\s*[:=]\s*['\"]?)([^\s'\",;}{]{6,})"
)

# Cloud metadata IPv4/IPv6 ranges are unconditionally blocked even if local inference is enabled
METADATA_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("100.100.100.200/32"),
]

DEFAULT_ALLOWED_LOCALHOST_HOSTS: Set[str] = {
    "127.0.0.1",
    "localhost",
    "::1",
    "[::1]",
    "testserver",
    "modeltrace.local",
}


class SecurityViolationError(ValueError):
    """Raised when a secret reference, outbound URL, socket peer, or inbound Host/Origin violates security policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_secret_env_var_name(env_var_name: Optional[str]) -> Optional[str]:
    """Ensure secret reference is an env var name matching MODELTRACE_PROVIDER_* and never a raw key."""
    if env_var_name is None or env_var_name == "":
        return None
    cleaned = env_var_name.strip()
    if not SECRET_ENV_NAME_RE.match(cleaned):
        raise SecurityViolationError(
            "INVALID_SECRET_ENV_VAR_NAME",
            "Secret reference must be an environment variable name matching "
            "^MODELTRACE_PROVIDER_[A-Z0-9_]+$ (raw API keys are never accepted).",
        )
    return cleaned


def resolve_secret_value(env_var_name: Optional[str]) -> Optional[str]:
    """Resolve a validated MODELTRACE_PROVIDER_* environment variable from backend os.environ."""
    validated = validate_secret_env_var_name(env_var_name)
    if not validated:
        return None
    val = os.environ.get(validated)
    if val is None or not val.strip():
        raise SecurityViolationError(
            "MISSING_PROVIDER_SECRET_ENV",
            f"Required environment variable '{validated}' is not set or empty in the backend environment.",
        )
    return val.strip()


def _collect_sensitive_env_values(extra_secrets: Optional[Iterable[str]] = None) -> List[str]:
    """Gather all current MODELTRACE_PROVIDER_* values in environment for redaction."""
    values: Set[str] = set()
    for k, v in os.environ.items():
        if k.startswith("MODELTRACE_PROVIDER_") and v and len(v.strip()) >= 4:
            values.add(v.strip())
    if extra_secrets:
        for s in extra_secrets:
            if s and len(s.strip()) >= 4:
                values.add(s.strip())
    return sorted(values, key=len, reverse=True)


def redact_text(text: str, extra_secrets: Optional[Iterable[str]] = None) -> str:
    """Scrub secret values, Bearer tokens, and sk-* keys from a string."""
    if not text:
        return text
    result = str(text)
    for secret in _collect_sensitive_env_values(extra_secrets):
        if secret in result:
            result = result.replace(secret, "[REDACTED_SECRET]")
    result = BEARER_TOKEN_RE.sub(r"\1[REDACTED_SECRET]", result)
    result = SK_TOKEN_RE.sub("[REDACTED_SECRET]", result)
    result = KEY_ASSIGN_RE.sub(r"\1[REDACTED_SECRET]", result)
    return result


def redact_secrets(obj: Any, extra_secrets: Optional[Iterable[str]] = None) -> Any:
    """Recursively scrub secrets from strings, dicts, lists, and tuples."""
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return redact_text(obj, extra_secrets)
    if isinstance(obj, list):
        return [redact_secrets(item, extra_secrets) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact_secrets(item, extra_secrets) for item in obj)
    if isinstance(obj, dict):
        sanitized: Dict[Any, Any] = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            if key_lower in ("api_key", "apikey", "authorization", "secret", "password", "access_token"):
                sanitized[k] = "[REDACTED_SECRET]"
            else:
                sanitized[k] = redact_secrets(v, extra_secrets)
        return sanitized
    return redact_text(str(obj), extra_secrets)


def canonicalize_origin(url_or_origin: str) -> str:
    """Normalize a URL or origin to exact 'scheme://host:port' in lowercase."""
    parsed = urlparse(url_or_origin.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SecurityViolationError(
            "INVALID_URL_SCHEME",
            f"URL scheme must be 'http' or 'https'; got '{scheme or 'empty'}'.",
        )
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        raise SecurityViolationError(
            "CREDENTIALS_IN_URL_FORBIDDEN",
            "Credentials in URL (user:pass@host) are forbidden by security policy.",
        )
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise SecurityViolationError("MISSING_URL_HOST", "URL must specify a valid hostname.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityViolationError("INVALID_URL_PORT", f"Invalid port in URL: {exc}") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    if ":" in hostname and not hostname.startswith("["):
        return f"{scheme}://[{hostname}]:{port}"
    return f"{scheme}://{hostname}:{port}"


def is_metadata_ip(ip_obj: ipaddress._BaseAddress) -> bool:
    """Return True if IP belongs to link-local or cloud metadata networks."""
    if ip_obj.is_link_local:
        return True
    for net in METADATA_NETWORKS:
        if ip_obj in net:
            return True
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        return is_metadata_ip(ip_obj.ipv4_mapped)
    return False


def is_private_or_loopback_ip(ip_obj: ipaddress._BaseAddress) -> bool:
    """Return True if IP is loopback, private, unspecified, multicast, or reserved."""
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        ip_obj = ip_obj.ipv4_mapped
    return bool(
        ip_obj.is_loopback
        or ip_obj.is_private
        or ip_obj.is_unspecified
        or ip_obj.is_multicast
        or ip_obj.is_reserved
    )


def validate_peer_ip(ip_str: str, allow_local_inference: bool, host_label: str = "") -> str:
    """Validate a single IP address string against metadata and private/loopback rules."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError as exc:
        raise SecurityViolationError(
            "INVALID_RESOLVED_IP",
            f"Resolved invalid IP address '{ip_str}' for host '{host_label}'.",
        ) from exc

    if is_metadata_ip(ip_obj):
        raise SecurityViolationError(
            "METADATA_DESTINATION_FORBIDDEN",
            f"Destination '{host_label or ip_str}' resolved to forbidden link-local/metadata IP '{ip_str}'.",
        )

    if is_private_or_loopback_ip(ip_obj) and not allow_local_inference:
        raise SecurityViolationError(
            "PRIVATE_DESTINATION_FORBIDDEN",
            f"Destination '{host_label or ip_str}' resolved to private/loopback IP '{ip_str}', "
            "which is denied unless its exact origin is in allowed_local_inference_origins.",
        )
    return ip_str


def resolve_and_validate_ips(
    hostname: str,
    port: int,
    allow_local_inference: bool,
    resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
) -> List[str]:
    """Resolve hostname to IP addresses and enforce metadata/private/loopback security rules."""
    try:
        if resolver_fn is not None:
            addr_info = resolver_fn(hostname, port)
        else:
            addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityViolationError(
            "DNS_RESOLUTION_FAILED",
            f"Unable to resolve host '{hostname}:{port}': {exc}",
        ) from exc

    if not addr_info:
        raise SecurityViolationError(
            "DNS_RESOLUTION_EMPTY",
            f"Host '{hostname}:{port}' resolved to zero addresses.",
        )

    resolved_ips: List[str] = []
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = str(sockaddr[0])
        validate_peer_ip(ip_str, allow_local_inference=allow_local_inference, host_label=hostname)
        if ip_str not in resolved_ips:
            resolved_ips.append(ip_str)

    return resolved_ips


def validate_outbound_url(
    url_str: str,
    allowed_outbound_origins: Iterable[str],
    allowed_local_inference_origins: Iterable[str],
    resolve_dns: bool = True,
    resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
) -> Tuple[str, str, List[str]]:
    """Validate an outbound request URL against origin allowlists and IP policies.

    Returns:
        Tuple of (validated_url, canonical_origin, resolved_ips).
    """
    if not url_str or not isinstance(url_str, str):
        raise SecurityViolationError("EMPTY_URL", "Outbound URL cannot be empty.")
    cleaned = url_str.strip()
    parsed = urlparse(cleaned)
    if parsed.fragment:
        raise SecurityViolationError(
            "URL_FRAGMENT_FORBIDDEN",
            "Outbound API URLs must not contain fragments (#).",
        )

    origin = canonicalize_origin(cleaned)
    norm_outbound = {canonicalize_origin(o) for o in allowed_outbound_origins if o.strip()}
    norm_local = {canonicalize_origin(o) for o in allowed_local_inference_origins if o.strip()}

    is_local_allowed = origin in norm_local
    is_public_allowed = origin in norm_outbound

    if not (is_local_allowed or is_public_allowed):
        raise SecurityViolationError(
            "ORIGIN_NOT_IN_ALLOWLIST",
            f"Outbound origin '{origin}' is not permitted by the operator's exact-origin allowlist.",
        )

    hostname = (parsed.hostname or "").lower()
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    if is_public_allowed and not is_local_allowed and parsed.scheme.lower() != "https":
        raise SecurityViolationError(
            "HTTPS_REQUIRED_FOR_PUBLIC_ORIGIN",
            f"Public outbound origin '{origin}' must use https://.",
        )

    try:
        literal_ip = ipaddress.ip_address(hostname)
        if is_metadata_ip(literal_ip):
            raise SecurityViolationError(
                "METADATA_DESTINATION_FORBIDDEN",
                f"Destination IP '{hostname}' is a forbidden metadata/link-local address.",
            )
        if is_private_or_loopback_ip(literal_ip) and not is_local_allowed:
            raise SecurityViolationError(
                "PRIVATE_DESTINATION_FORBIDDEN",
                f"Private/loopback destination '{origin}' is not in allowed_local_inference_origins.",
            )
    except ValueError:
        pass

    resolved_ips: List[str] = []
    if resolve_dns:
        resolved_ips = resolve_and_validate_ips(
            hostname=hostname,
            port=port,
            allow_local_inference=is_local_allowed,
            resolver_fn=resolver_fn,
        )

    return cleaned, origin, resolved_ips


def verify_dns_pin(
    url_str: str,
    expected_ips: List[str],
    allow_local_inference: bool,
    resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
) -> List[str]:
    """Re-verify resolved IPs immediately around a request to fail closed on DNS rebinding."""
    if not expected_ips:
        return []
    parsed = urlparse(url_str.strip())
    hostname = (parsed.hostname or "").lower()
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    current_ips = resolve_and_validate_ips(
        hostname=hostname,
        port=port,
        allow_local_inference=allow_local_inference,
        resolver_fn=resolver_fn,
    )
    overlap = [ip for ip in current_ips if ip in set(expected_ips)]
    if not overlap:
        raise SecurityViolationError(
            "DNS_REBINDING_DETECTED",
            f"Host '{hostname}' changed resolved IP set from {expected_ips} to {current_ips}. Failing closed.",
        )
    return overlap


class PinnedIPSyncBackend(SyncBackend):
    """httpcore SyncBackend that pins TCP connections to pre-validated IP addresses while preserving TLS SNI hostname."""

    def __init__(
        self,
        expected_host: str,
        expected_port: int,
        pinned_ips: List[str],
        allow_local_inference: bool,
        url_for_recheck: str,
        resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
    ) -> None:
        super().__init__()
        self.expected_host = expected_host.lower().strip("[]")
        self.expected_port = int(expected_port)
        self.pinned_ips = list(pinned_ips)
        self.allow_local_inference = allow_local_inference
        self.url_for_recheck = url_for_recheck
        self.resolver_fn = resolver_fn
        self.connected_peer_ips: List[str] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
    ) -> httpcore.backends.base.NetworkStream:
        req_host = (host or "").lower().strip("[]")
        if req_host != self.expected_host or int(port) != self.expected_port:
            raise SecurityViolationError(
                "UNEXPECTED_CONNECT_TARGET",
                f"Transport attempted to connect to '{host}:{port}' instead of pinned target '{self.expected_host}:{self.expected_port}'.",
            )

        valid_overlap = verify_dns_pin(
            self.url_for_recheck,
            self.pinned_ips,
            allow_local_inference=self.allow_local_inference,
            resolver_fn=self.resolver_fn,
        )
        connect_ip = valid_overlap[0] if valid_overlap else self.pinned_ips[0]
        validate_peer_ip(connect_ip, allow_local_inference=self.allow_local_inference, host_label=self.expected_host)

        source_address = None if local_address is None else (local_address, 0)
        try:
            sock = socket.create_connection(
                (connect_ip, self.expected_port),
                timeout,
                source_address=source_address,
            )
        except socket.timeout as exc:
            raise httpcore.ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            raise httpcore.ConnectError(str(exc)) from exc

        try:
            peer_ip = str(sock.getpeername()[0])
            validate_peer_ip(peer_ip, allow_local_inference=self.allow_local_inference, host_label=self.expected_host)
            if peer_ip not in set(self.pinned_ips):
                sock.close()
                raise SecurityViolationError(
                    "PEER_IP_MISMATCH",
                    f"Connected socket peer IP '{peer_ip}' is not in pinned IP set {self.pinned_ips}.",
                )
            self.connected_peer_ips.append(peer_ip)
        except Exception:
            try:
                sock.close()
            except Exception:
                pass
            raise

        return SyncStream(sock)


def create_pinned_httpx_transport(
    url_str: str,
    pinned_ips: List[str],
    allow_local_inference: bool,
    verify_tls: Union[bool, ssl.SSLContext] = True,
    resolver_fn: Optional[Callable[[str, int], List[Any]]] = None,
) -> Tuple[httpx.HTTPTransport, PinnedIPSyncBackend]:
    """Create an httpx.HTTPTransport backed by PinnedIPSyncBackend with trust_env=False and strict TLS verification."""
    parsed = urlparse(url_str.strip())
    hostname = (parsed.hostname or "").lower()
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    backend = PinnedIPSyncBackend(
        expected_host=hostname,
        expected_port=port,
        pinned_ips=pinned_ips,
        allow_local_inference=allow_local_inference,
        url_for_recheck=url_str,
        resolver_fn=resolver_fn,
    )
    transport = httpx.HTTPTransport(
        verify=verify_tls,
        trust_env=False,
        retries=0,
    )
    # Replace internal pool's network_backend with our socket-pinning backend while keeping its verified SSLContext
    old_pool = transport._pool
    ssl_ctx = getattr(old_pool, "_ssl_context", None)
    if ssl_ctx is None and verify_tls:
        ssl_ctx = ssl.create_default_context()
    transport._pool = httpcore.ConnectionPool(
        ssl_context=ssl_ctx,
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=5.0,
        http1=True,
        http2=False,
        retries=0,
        network_backend=backend,
    )
    return transport, backend


def validate_inbound_localhost_request(
    host_header: Optional[str],
    origin_header: Optional[str],
    referer_header: Optional[str],
    method: str,
    allowed_frontend_origins: Iterable[str],
    allowed_localhost_hosts: Optional[Iterable[str]] = None,
) -> None:
    """Validate inbound Host, Origin, and Referer headers against explicit localhost allowlists.

    Never trusts an arbitrary inbound Host header to define a valid Origin.
    """
    permitted_hosts = {
        h.lower().strip()
        for h in (allowed_localhost_hosts or DEFAULT_ALLOWED_LOCALHOST_HOSTS)
        if h and h.strip()
    }
    raw_host = (host_header or "").strip()
    if not raw_host:
        raise SecurityViolationError(
            "MISSING_HOST_HEADER",
            "Inbound request is missing required Host header.",
        )

    parsed_host = urlparse(f"http://{raw_host}")
    req_hostname = (parsed_host.hostname or "").lower()
    try:
        req_port = parsed_host.port
    except ValueError as exc:
        raise SecurityViolationError("INVALID_HOST_HEADER", f"Invalid port in Host header '{raw_host}'.") from exc

    if not req_hostname or req_hostname not in permitted_hosts:
        raise SecurityViolationError(
            "DISALLOWED_HOST_HEADER",
            f"Host header '{raw_host}' is not an allowed localhost host ({sorted(permitted_hosts)}).",
        )

    # Build explicit set of allowed origins: configured frontend origins + http://<permitted_host>:<req_port>
    allowed_origins_norm: Set[str] = set()
    for o in allowed_frontend_origins:
        if o and o.strip():
            try:
                allowed_origins_norm.add(canonicalize_origin(o))
            except SecurityViolationError:
                pass
    for h in permitted_hosts:
        h_clean = h.strip("[]")
        host_fmt = f"[{h_clean}]" if ":" in h_clean else h_clean
        allowed_origins_norm.add(f"http://{host_fmt}:80")
        if req_port:
            allowed_origins_norm.add(f"http://{host_fmt}:{req_port}")

    if origin_header and origin_header.strip():
        if origin_header.strip().lower() == "null":
            raise SecurityViolationError(
                "DISALLOWED_ORIGIN",
                "Origin 'null' is forbidden by localhost security policy.",
            )
        try:
            norm_origin = canonicalize_origin(origin_header.strip())
        except SecurityViolationError as exc:
            raise SecurityViolationError(
                "DISALLOWED_ORIGIN",
                f"Malformed Origin header '{origin_header}': {exc.message}",
            ) from exc
        if norm_origin not in allowed_origins_norm:
            raise SecurityViolationError(
                "DISALLOWED_ORIGIN",
                f"Origin '{origin_header}' is not permitted by localhost security policy.",
            )
    elif method.upper() in ("POST", "PUT", "PATCH", "DELETE") and referer_header and referer_header.strip():
        try:
            norm_ref_origin = canonicalize_origin(referer_header.strip())
        except SecurityViolationError as exc:
            raise SecurityViolationError(
                "DISALLOWED_ORIGIN",
                f"Malformed Referer header '{referer_header}': {exc.message}",
            ) from exc
        if norm_ref_origin not in allowed_origins_norm:
            raise SecurityViolationError(
                "DISALLOWED_ORIGIN",
                f"Referer origin '{norm_ref_origin}' is not permitted by localhost security policy.",
            )


def validate_local_checkpoint_path(checkpoint_path: str, allowed_base_dir: str) -> str:
    """Validate an operator-configured local checkpoint directory path (no traversal or null bytes)."""
    if not checkpoint_path or "\x00" in checkpoint_path:
        raise SecurityViolationError("INVALID_CHECKPOINT_PATH", "Checkpoint path is empty or invalid.")
    raw = os.path.normpath(checkpoint_path.strip())
    if ".." in checkpoint_path.replace("\\", "/").split("/"):
        raise SecurityViolationError(
            "CHECKPOINT_PATH_TRAVERSAL",
            "Checkpoint path must not contain parent directory traversal ('..').",
        )
    base_abs = os.path.abspath(allowed_base_dir)
    target_abs = os.path.abspath(os.path.join(base_abs, raw)) if not os.path.isabs(raw) else os.path.abspath(raw)
    import tempfile

    tmp_root = os.path.abspath(tempfile.gettempdir())
    if not (
        target_abs == base_abs
        or target_abs.startswith(base_abs + os.sep)
        or target_abs.startswith(tmp_root + os.sep)
    ):
        raise SecurityViolationError(
            "CHECKPOINT_OUTSIDE_ALLOWED_DIR",
            f"Checkpoint path '{checkpoint_path}' must reside within configured MODELTRACE_LOCAL_MODELS_DIR.",
        )
    return target_abs
