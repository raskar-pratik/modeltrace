"""ModelTrace V1 automated test suite package."""

import importlib.util
from pathlib import Path
import sys

_TESTS_DIR = Path(__file__).resolve().parent
_KNOWN_MODULES = [
    "test_http_adapter_and_worker",
    "test_local_transformers_and_api_cli",
    "test_migrations_and_db",
    "test_security_and_redaction",
    "test_statistics_and_verdicts",
    "test_ui_e2e_browser",
]


def load_tests(loader, standard_tests, pattern):
    suite = loader.suiteClass()
    seen = set()
    for mod_name in _KNOWN_MODULES:
        mod_path = _TESTS_DIR / f"{mod_name}.py"
        if not mod_path.is_file():
            continue
        full_name = f"tests.{mod_name}"
        if full_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_name, str(mod_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
        suite.addTests(loader.loadTestsFromModule(sys.modules[full_name]))
        seen.add(mod_name)
    return suite
