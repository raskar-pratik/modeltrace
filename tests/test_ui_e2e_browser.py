"""Production-browser workflow and setup failure regressions.

Set MODELTRACE_REQUIRE_BROWSER=1 in release CI to fail if a real built UI or browser is absent.
Browser binary paths can be supplied through MODELTRACE_CHROME_BINARY and MODELTRACE_CHROMEDRIVER_PATH.
Form edits use the native value setter so real React controlled inputs observe the change.
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional

import httpx
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from modeltrace.config import ROOT_DIR

SCREENSHOT_DIR = ROOT_DIR / "docs" / "screenshots"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def resolve_chrome_binary() -> Optional[str]:
    for env_key in ("MODELTRACE_CHROME_BINARY", "CHROME_BIN"):
        val = os.environ.get(env_key, "").strip()
        if val and Path(val).is_file():
            return val
    for candidate in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def resolve_chromedriver_path() -> Optional[str]:
    for env_key in ("MODELTRACE_CHROMEDRIVER_PATH", "CHROMEDRIVER_PATH"):
        val = os.environ.get(env_key, "").strip()
        if val and Path(val).is_file():
            return val
    return shutil.which("chromedriver")


class TestSetupFailFastAndBuildGuards(unittest.TestCase):
    def test_build_frontend_rejects_implicit_offline_fallback(self) -> None:
        """A disposable source checkout without dependencies must refuse any substitute build."""
        env = {k: v for k, v in os.environ.items() if k != "MODELTRACE_ALLOW_DEV_OFFLINE_FALLBACK"}
        with tempfile.TemporaryDirectory() as tmp:
            scripts = Path(tmp) / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT_DIR / "scripts" / "build_frontend.cjs", scripts / "build_frontend.cjs")
            res = subprocess.run(
                ["node", str(scripts / "build_frontend.cjs")],
                cwd=tmp, env=env, capture_output=True, text=True,
            )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Production frontend build requires installed project-local TypeScript", res.stderr)

    def test_setup_sh_fails_fast_on_dependency_or_typecheck_failure(self) -> None:
        """Intentionally failing pip or npm/typecheck commands in setup.sh must abort immediately and never print Setup complete."""
        if os.name == "nt":
            self.skipTest("POSIX shell setup is verified on Linux; Windows uses setup.ps1.")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            scripts_dir = tmp_root / "scripts"
            scripts_dir.mkdir(parents=True)
            shutil.copy2(ROOT_DIR / "scripts" / "setup.sh", scripts_dir / "setup.sh")
            (tmp_root / "requirements.lock").write_text("numpy==1.26.4\n", encoding="utf-8")
            (tmp_root / "frontend").mkdir()
            (tmp_root / "frontend" / "package.json").write_text('{"name": "frontend"}\n', encoding="utf-8")

            # Case A: Mock `python3` so `pip install -r requirements.lock` fails with exit code 1
            fake_bin_a = tmp_root / "fake_bin_a"
            fake_bin_a.mkdir()
            fake_py_a = fake_bin_a / "python3"
            fake_py_a.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
                "  mkdir -p \"$3/bin\"\n"
                "  cat << 'EOF' > \"$3/bin/python3\"\n"
                "#!/usr/bin/env bash\n"
                'if [[ "$*" == *"install -r requirements.lock"* ]]; then\n'
                '  echo "Simulated pip lockfile failure" >&2\n'
                "  exit 42\n"
                "fi\n"
                "exit 0\n"
                "EOF\n"
                "  chmod +x \"$3/bin/python3\"\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_py_a.chmod(0o755)
            fake_npm_a = fake_bin_a / "npm"
            fake_npm_a.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_npm_a.chmod(0o755)

            env_a = os.environ.copy()
            env_a["PATH"] = f"{fake_bin_a}:{env_a.get('PATH', '')}"
            res_a = subprocess.run(
                ["bash", str(scripts_dir / "setup.sh")],
                cwd=str(tmp_root),
                env=env_a,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_a.returncode, 42)
            self.assertNotIn("Setup complete", res_a.stdout + res_a.stderr)

            # Case B: Mock `npm` so `npm --prefix frontend run typecheck` fails with exit code 2
            fake_bin_b = tmp_root / "fake_bin_b"
            fake_bin_b.mkdir()
            shutil.rmtree(tmp_root / ".venv", ignore_errors=True)
            fake_py_b = fake_bin_b / "python3"
            fake_py_b.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
                "  mkdir -p \"$3/bin\"\n"
                "  cat << 'EOF' > \"$3/bin/python3\"\n"
                "#!/usr/bin/env bash\n"
                "exit 0\n"
                "EOF\n"
                "  cat << 'EOF' > \"$3/bin/modeltrace\"\n"
                "#!/usr/bin/env bash\n"
                "exit 0\n"
                "EOF\n"
                "  chmod +x \"$3/bin/python3\" \"$3/bin/modeltrace\"\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_py_b.chmod(0o755)
            fake_npm_b = fake_bin_b / "npm"
            fake_npm_b.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$*" == *"run typecheck"* ]]; then\n'
                '  echo "error TS2322: Type string is not assignable to type number." >&2\n'
                "  exit 2\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_npm_b.chmod(0o755)

            env_b = os.environ.copy()
            env_b["PATH"] = f"{fake_bin_b}:{env_b.get('PATH', '')}"
            res_b = subprocess.run(
                ["bash", str(scripts_dir / "setup.sh")],
                cwd=str(tmp_root),
                env=env_b,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_b.returncode, 2)
            self.assertIn("TS2322", res_b.stderr)
            self.assertNotIn("Setup complete", res_b.stdout + res_b.stderr)

    def test_setup_ps1_enforces_exit_code_checks_on_every_native_step(self) -> None:
        """Verify setup.ps1 wraps every native command in Invoke-NativeStep and checks $LASTEXITCODE -ne 0."""
        ps1_path = ROOT_DIR / "scripts" / "setup.ps1"
        content = ps1_path.read_text(encoding="utf-8")
        self.assertIn("function Invoke-NativeStep", content)
        self.assertIn("if ($LASTEXITCODE -ne 0)", content)
        self.assertIn("exit $LASTEXITCODE", content)
        self.assertNotIn("build_frontend.cjs", content)
        self.assertNotIn("--allow-dev-offline-fallback", content)

        # Ensure every python, pip, npm, and modeltrace invocation goes through Invoke-NativeStep
        step_calls = re.findall(r"Invoke-NativeStep\s+\"([^\"]+)\"", content)
        self.assertGreaterEqual(len(step_calls), 7)
        joined_steps = " | ".join(step_calls)
        self.assertIn("requirements.lock", joined_steps)
        self.assertIn("tsc --noEmit", joined_steps)
        self.assertIn("vite build", joined_steps)
        self.assertIn("modeltrace migrate", joined_steps)

    @unittest.skipUnless(shutil.which("pwsh"), "Native PowerShell is not installed on this runner.")
    def test_native_powershell_stops_after_failed_command(self) -> None:
        content = (ROOT_DIR / "scripts/setup.ps1").read_text(encoding="utf-8")
        guard = content.split("$RootDir =", 1)[0]
        command = guard + '\nInvoke-NativeStep "negative control" { & "' + sys.executable + '" -c "import sys;sys.exit(42)" }; Write-Host "Setup complete"'
        result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-Command", command], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Setup complete", result.stdout)
        self.assertIn("42", result.stderr)


class TestBrowserEndToEndFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dist_index = ROOT_DIR / "frontend" / "dist" / "index.html"
        if not dist_index.is_file():
            if os.environ.get("MODELTRACE_REQUIRE_BROWSER") == "1":
                raise RuntimeError("Browser verification requires a built production frontend.")
            raise unittest.SkipTest("Run npm --prefix frontend run build before browser verification.")

        chrome_bin = resolve_chrome_binary()
        chromedriver_bin = resolve_chromedriver_path()
        if not chrome_bin:
            if os.environ.get("MODELTRACE_REQUIRE_BROWSER") == "1":
                raise RuntimeError("Required Chromium/Chrome binary was not provisioned.")
            raise unittest.SkipTest("Skipping browser E2E: Chromium/Chrome binary not found on PATH.")

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp_dir.cleanup)
        cls.db_path = str(Path(cls.tmp_dir.name) / "e2e_browser.db")
        cls.exports_dir = str(Path(cls.tmp_dir.name) / "exports")
        cls.port = _find_free_port()
        cls.api_url = f"http://127.0.0.1:{cls.port}"
        cls.browser_url = f"http://modeltrace.local:{cls.port}"

        env = os.environ.copy()
        env["MODELTRACE_DB_PATH"] = cls.db_path
        env["MODELTRACE_EXPORTS_DIR"] = cls.exports_dir

        cls.server_proc = subprocess.Popen(
            [sys.executable, "-m", "modeltrace.cli", "serve", "--host", "127.0.0.1", "--port", str(cls.port)],
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cls.addClassCleanup(cls._stop_server)

        ready = False
        for _ in range(40):
            try:
                r = httpx.get(f"{cls.api_url}/api/health", timeout=1.0, trust_env=False)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.25)
        if not ready:
            cls.server_proc.terminate()
            raise RuntimeError("Live uvicorn server failed to start for E2E browser test.")

        opts = Options()
        opts.binary_location = chrome_bin
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-proxy-server")
        opts.add_argument("--host-resolver-rules=MAP modeltrace.local 127.0.0.1")
        opts.add_argument("--window-size=1440,960")
        opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        if chromedriver_bin:
            cls.driver = webdriver.Chrome(service=Service(chromedriver_bin), options=opts)
        else:
            cls.driver = webdriver.Chrome(options=opts)
        cls.addClassCleanup(cls.driver.quit)

    @classmethod
    def _stop_server(cls) -> None:
        if hasattr(cls, "server_proc"):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(cls.server_proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                cls.server_proc.terminate()
            try:
                cls.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_proc.kill()

    def _wait_for_testid(self, testid: str, timeout: float = 8.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            els = self.driver.find_elements(By.CSS_SELECTOR, f'[data-testid="{testid}"]')
            if els:
                return els[0]
            time.sleep(0.15)
        raise AssertionError(f"Timed out waiting for [data-testid='{testid}']")

    def _set_input_value(self, element, value: str) -> None:
        self.driver.execute_script(
            """
            const el = arguments[0];
            const val = arguments[1];
            const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
            setter.call(el, val);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            """,
            element,
            value,
        )

    def _assert_no_horizontal_overflow(self, label: str) -> None:
        overflow = self.driver.execute_script(
            "return document.documentElement.scrollWidth > window.innerWidth + 2;"
        )
        self.assertFalse(overflow, f"Horizontal page overflow detected on {label}")

    def test_full_connected_browser_workflow_and_responsive_layouts(self) -> None:
        d = self.driver
        d.set_window_size(1440, 960)
        d.get(self.browser_url)

        # 1. Overview Empty State initially
        new_audit_btn = self._wait_for_testid("overview-new-audit-btn")
        self.assertIsNotNone(new_audit_btn)
        brand_mark = d.find_element(By.CSS_SELECTOR, 'img[src="/modeltrace-mark.svg"]')
        WebDriverWait(d, 8).until(lambda driver: driver.execute_script("return arguments[0].complete && arguments[0].naturalWidth > 0;", brand_mark))
        self._assert_no_horizontal_overflow("overview-empty-1440")

        # 2. Navigate to Endpoint Profiles -> test recoverable form error + valid profile + adapter failure
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-profiles"]').click()
        name_inp = self._wait_for_testid("profile-name-input")
        self._set_input_value(name_inp, "Unreachable Local Test Target")

        url_inp = d.find_element(By.CSS_SELECTOR, '[data-testid="profile-url-input"]')
        self._set_input_value(url_inp, "http://127.0.0.1:11434/v1")

        # Enter an invalid raw secret name first to verify form preservation on error
        env_inp = d.find_element(By.CSS_SELECTOR, '[data-testid="profile-env-input"]')
        self._set_input_value(env_inp, "sk-raw-secret-forbidden")
        d.find_element(By.CSS_SELECTOR, '[data-testid="profile-submit-btn"]').click()

        err_banner = self._wait_for_testid("profile-error-banner")
        self.assertIn("MODELTRACE_PROVIDER_", err_banner.text)
        self.assertEqual(
            d.find_element(By.CSS_SELECTOR, '[data-testid="profile-name-input"]').get_attribute("value"),
            "Unreachable Local Test Target",
        )

        # Fix env var to valid MODELTRACE_PROVIDER_* name and submit
        env_inp = d.find_element(By.CSS_SELECTOR, '[data-testid="profile-env-input"]')
        self._set_input_value(env_inp, "MODELTRACE_PROVIDER_UNSET_TEST_KEY")
        d.find_element(By.CSS_SELECTOR, '[data-testid="profile-submit-btn"]').click()
        succ_banner = self._wait_for_testid("profile-success-banner")
        self.assertIn("Created profile", succ_banner.text)

        # Trigger real bounded connection test on the newly created profile -> triggers MISSING_PROVIDER_SECRET_ENV adapter failure
        profile_id_match = re.search(r"\((prof_[a-z0-9]+)\)", succ_banner.text)
        self.assertIsNotNone(profile_id_match, "Created profile banner must identify the saved profile")
        profile_id = profile_id_match.group(1)
        self._wait_for_testid(f"test-conn-btn-{profile_id}").click()
        WebDriverWait(d, 8).until(lambda driver: "MISSING_PROVIDER_SECRET_ENV" in driver.page_source)
        self.assertIn("MISSING_PROVIDER_SECRET_ENV", d.page_source)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_profiles.png"))
        self._assert_no_horizontal_overflow("profiles-1440")

        # 3. Navigate to References & Calibration -> check missing HF prerequisites
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-references"]').click()
        _cal_btn = self._wait_for_testid("run-demo-calibration-btn")
        self.assertIn("torch", d.page_source)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_references.png"))

        # 4. Navigate to Probe Library -> import custom JSONL probes & edit into new immutable revision
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-probes"]').click()
        imp_name = self._wait_for_testid("probe-import-name-input")
        self._set_input_value(imp_name, "E2E Imported Probe Suite")
        imp_txt = d.find_element(By.CSS_SELECTOR, '[data-testid="probe-import-content-textarea"]')
        self._set_input_value(
            imp_txt,
            '{"prompt_id": "e2e-01", "family": "reasoning", "title": "Prime Check", "prompt_text": "Is 97 prime?"}\n'
            '{"prompt_id": "e2e-02", "family": "code", "title": "Reverse String", "prompt_text": "Write reverse(s)."}\n'
            '{"prompt_id": "e2e-03", "family": "extraction", "title": "Date Parse", "prompt_text": "Extract year from 2026-10-14."}\n'
            '{"prompt_id": "e2e-04", "family": "formatting", "title": "JSON Pair", "prompt_text": "Output {\\"ok\\":true}."}',
        )
        d.find_element(By.CSS_SELECTOR, '[data-testid="probe-import-submit-btn"]').click()
        probe_succ = self._wait_for_testid("probe-success-banner")
        self.assertIn("Imported 4 prompts", probe_succ.text)

        # Edit prompt e2e-01 into new immutable revision (#2)
        edit_btn = self._wait_for_testid("edit-prompt-btn-e2e-01")
        edit_btn.click()
        edit_area = self._wait_for_testid("edit-prompt-textarea")
        self._set_input_value(
            edit_area,
            "Determine whether 97 is a prime number and explain why in one sentence.",
        )
        d.find_element(By.CSS_SELECTOR, '[data-testid="save-new-revision-btn"]').click()
        probe_succ2 = self._wait_for_testid("probe-success-banner")
        self.assertIn("Created new immutable revision #2", probe_succ2.text)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_probes.png"))

        # 5. Navigate to New Audit -> preview, freeze-only & cancel, then run Synthetic DEMO audit
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-new-audit"]').click()
        n_inp = self._wait_for_testid("new-audit-nprompts-input")
        self._set_input_value(n_inp, "4")
        m_inp = d.find_element(By.CSS_SELECTOR, '[data-testid="new-audit-mref-input"]')
        self._set_input_value(m_inp, "4")
        time.sleep(0.5)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_new_audit.png"))

        # First freeze an audit in CREATED state without starting, then cancel it
        d.find_element(By.CSS_SELECTOR, '[data-testid="freeze-only-audit-btn"]').click()
        cancel_btn = self._wait_for_testid("audit-detail-cancel-btn")
        cancel_btn.click()
        notice = self._wait_for_testid("audit-action-notice")
        self.assertIn("Requested cancellation", notice.text)

        # Now go back to New Audit and launch a real Synthetic DEMO Substituted audit
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-new-audit"]').click()
        n_inp2 = self._wait_for_testid("new-audit-nprompts-input")
        self._set_input_value(n_inp2, "4")
        m_inp2 = d.find_element(By.CSS_SELECTOR, '[data-testid="new-audit-mref-input"]')
        self._set_input_value(m_inp2, "4")
        demo_sub_btn = self._wait_for_testid("quick-demo-substituted-btn")
        demo_sub_btn.click()

        # Wait for worker to finish the 4-prompt DEMO audit on the Audit History screen
        deadline = time.time() + 15.0
        completed_found = False
        while time.time() < deadline:
            reason_el = self._wait_for_testid("audit-detail-verdict-reason")
            if "rejects the null hypothesis" in reason_el.text:
                completed_found = True
                break
            time.sleep(0.4)
        self.assertTrue(completed_found, "DEMO audit did not complete within 15s")
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_audits.png"))

        # 6. Navigate to Evidence Explorer -> inspect real-data SVG charts & stored responses
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-evidence"]').click()
        time.sleep(0.6)
        self.assertIn("OBSERVED RANK DISTRIBUTION", d.page_source)
        self.assertIn("STORED RAW RESPONSES", d.page_source)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_evidence.png"))
        self._assert_no_horizontal_overflow("evidence-1440")

        # 7. Reload page on /audits to verify reload preserves jobs
        d.get(f"{self.browser_url}/audits")
        self._wait_for_testid("audit-detail-verdict-reason")
        self.assertIn("[DEMO] Synthetic Audit (substituted)", d.page_source)

        # 8. Navigate to Reports & Settings -> verify export links & confirm Evidence Purge
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-reports"]').click()
        html_link = self._wait_for_testid("export-html-link")
        href = html_link.get_attribute("href") or ""
        self.assertIn("/export?format=html", href)
        export_path = "/" + href.split("/", 3)[-1]
        r_exp = httpx.get(f"{self.api_url}{export_path}", timeout=5.0, trust_env=False)
        self.assertEqual(r_exp.status_code, 200)
        self.assertIn("Know what your API is serving.", r_exp.text)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_reports_settings.png"))

        # Open purge confirmation modal and confirm evidence deletion while preserving manifest
        d.find_element(By.CSS_SELECTOR, '[data-testid="open-purge-modal-btn"]').click()
        confirm_btn = self._wait_for_testid("confirm-purge-evidence-btn")
        confirm_btn.click()
        status_banner = self._wait_for_testid("settings-status-banner")
        self.assertIn("Historical manifest and statistical summary are preserved", status_banner.text)

        # 9. Navigate back to Overview -> capture 1440px, 768px, and 390px responsive screenshots
        d.find_element(By.CSS_SELECTOR, '[data-testid="nav-tab-overview"]').click()
        self._wait_for_testid("overview-latest-audit-name")
        time.sleep(0.4)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_1440_overview.png"))
        self._assert_no_horizontal_overflow("overview-1440")

        # Tablet 768px
        d.set_window_size(768, 1024)
        time.sleep(0.3)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_768_overview.png"))
        self._assert_no_horizontal_overflow("overview-768")

        # Mobile 390px
        d.set_window_size(390, 844)
        time.sleep(0.3)
        d.save_screenshot(str(SCREENSHOT_DIR / "ui_390_overview.png"))
        self._assert_no_horizontal_overflow("overview-390")

        # Verify zero severe JS runtime exceptions in browser console
        logs = d.get_log("browser")
        severe_js_errors = [
            entry
            for entry in logs
            if entry.get("level") == "SEVERE"
            and entry.get("source") == "javascript"
        ]
        self.assertEqual(
            severe_js_errors,
            [],
            f"Unexpected browser JS runtime errors: {severe_js_errors}",
        )


if __name__ == "__main__":
    unittest.main()
