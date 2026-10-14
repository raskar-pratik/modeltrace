"""Central configuration and deterministic helpers for ModelTrace."""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, List
from dotenv import load_dotenv

APP_NAME = "ModelTrace"
APP_TAGLINE = "Know what your API is serving."
ROOT_DIR = Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for SHA-256 integrity hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: Any) -> str:
    """Compute SHA-256 hex digest of string, bytes, or JSON-serializable object."""
    if isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = canonical_json(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class Settings:
    """Runtime environment settings for ModelTrace."""

    def __init__(self) -> None:
        self.db_path: str = os.environ.get(
            "MODELTRACE_DB_PATH", str(ROOT_DIR / "data" / "modeltrace.db")
        )
        self.host: str = os.environ.get("MODELTRACE_HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("MODELTRACE_PORT", "8000"))
        self.local_models_dir: str = os.environ.get(
            "MODELTRACE_LOCAL_MODELS_DIR", str(ROOT_DIR / "data" / "local_models")
        )
        self.exports_dir: str = os.environ.get(
            "MODELTRACE_EXPORTS_DIR", str(ROOT_DIR / "exports")
        )
        raw_outbound = os.environ.get(
            "MODELTRACE_ALLOWED_OUTBOUND_ORIGINS",
            os.environ.get("MODELTRACE_DEFAULT_ALLOWED_OUTBOUND_ORIGINS", "https://api.openai.com:443,https://api.together.xyz:443"),
        )
        self.default_allowed_outbound_origins: List[str] = [
            o.strip() for o in raw_outbound.split(",") if o.strip()
        ]
        raw_local = os.environ.get(
            "MODELTRACE_ALLOWED_LOCAL_INFERENCE_ORIGINS",
            os.environ.get("MODELTRACE_DEFAULT_ALLOWED_LOCAL_ORIGINS", "http://127.0.0.1:11434,http://127.0.0.1:8080"),
        )
        self.default_allowed_local_origins: List[str] = [
            o.strip() for o in raw_local.split(",") if o.strip()
        ]
        self.allowed_frontend_origins: List[str] = [o.strip() for o in os.environ.get(
            "MODELTRACE_ALLOWED_FRONTEND_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000,http://testserver",
        ).split(",") if o.strip()]
        self.max_upload_bytes: int = int(os.environ.get("MODELTRACE_MAX_UPLOAD_BYTES", "2097152"))
        self.max_response_bytes: int = int(
            os.environ.get("MODELTRACE_MAX_RESPONSE_BYTES", str(1024 * 1024))
        )


def get_env_settings() -> Settings:
    """Load current environment settings."""
    load_dotenv(ROOT_DIR / ".env", override=False)
    return Settings()
