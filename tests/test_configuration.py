"""Exercise documented .env configuration through the real settings loader."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from modeltrace.config import get_env_settings


class TestConfiguration(unittest.TestCase):
    def test_dotenv_and_existing_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "MODELTRACE_PORT=9000\nMODELTRACE_MAX_UPLOAD_BYTES=4096\n"
                "MODELTRACE_ALLOWED_OUTBOUND_ORIGINS=https://provider.example:443\n"
                "MODELTRACE_ALLOWED_LOCAL_INFERENCE_ORIGINS=http://127.0.0.1:9999\n"
                "MODELTRACE_ALLOWED_FRONTEND_ORIGINS=http://localhost:9000\n"
            )
            with patch("modeltrace.config.ROOT_DIR", root), patch.dict(os.environ, {"MODELTRACE_PORT": "9001"}, clear=True):
                settings = get_env_settings()
                self.assertEqual(settings.port, 9001)
                self.assertEqual(settings.max_upload_bytes, 4096)
                self.assertEqual(settings.default_allowed_outbound_origins, ["https://provider.example:443"])
                self.assertEqual(settings.default_allowed_local_origins, ["http://127.0.0.1:9999"])
                self.assertEqual(settings.allowed_frontend_origins, ["http://localhost:9000"])

    def test_legacy_origin_names_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("modeltrace.config.ROOT_DIR", Path(directory)), patch.dict(os.environ, {
                "MODELTRACE_DEFAULT_ALLOWED_OUTBOUND_ORIGINS": "https://legacy.example:443",
                "MODELTRACE_DEFAULT_ALLOWED_LOCAL_ORIGINS": "http://localhost:8888",
            }, clear=True):
                settings = get_env_settings()
                self.assertEqual(settings.default_allowed_outbound_origins, ["https://legacy.example:443"])
                self.assertEqual(settings.default_allowed_local_origins, ["http://localhost:8888"])
