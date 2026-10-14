"""Standard library unittest runner for ModelTrace (`python3 -m modeltrace.test_runner`).

IMPORTANT: This runner executes standard library `unittest` test suites when the upstream
`pytest` package cannot be installed due to offline network restrictions (`BLOCKED`). It is
NOT upstream `pytest`; all `tests/test_*.py` modules are `unittest.TestCase` classes that run
under both `pytest` (in online/CI environments) and `python3 -m unittest` / `modeltrace.test_runner`.
"""

import importlib.util
import sys
import time
import unittest
from pathlib import Path

from modeltrace.config import ROOT_DIR

CORE_TEST_MODULES = [
    "tests.test_migrations_and_db",
    "tests.test_statistics_and_verdicts",
    "tests.test_security_and_redaction",
    "tests.test_http_adapter_and_worker",
    "tests.test_local_transformers_and_api_cli",
    "tests.test_ui_e2e_browser",
]


def _load_test_module(mod_name: str):
    try:
        return __import__(mod_name, fromlist=["*"])
    except ModuleNotFoundError:
        rel_path = ROOT_DIR / (mod_name.replace(".", "/") + ".py")
        if rel_path.is_file():
            spec = importlib.util.spec_from_file_location(mod_name, str(rel_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                return mod
        raise


def run_all_tests() -> int:
    t0 = time.perf_counter()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for mod_name in CORE_TEST_MODULES:
        mod = _load_test_module(mod_name)
        suite.addTests(loader.loadTestsFromModule(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.perf_counter() - t0
    total = result.testsRun
    skipped = len(result.skipped)
    failed = len(result.failures) + len(result.errors)
    passed = total - skipped - failed
    print(
        f"\n=== ModelTrace Unittest Summary (stdlib unittest; upstream pytest BLOCKED in offline VM): "
        f"{passed} passed, {skipped} skipped (NOT RUN), {failed} failed in {elapsed:.2f}s ==="
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
