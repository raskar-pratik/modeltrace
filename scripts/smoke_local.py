#!/usr/bin/env python3
"""Start a disposable local server and check production UI delivery. No provider calls."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory() as tmp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env["MODELTRACE_DB_PATH"] = str(Path(tmp) / "smoke.db")
        env["MODELTRACE_EXPORTS_DIR"] = str(Path(tmp) / "exports")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (Path(tmp) / "server.log").open("w+") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "modeltrace.cli", "serve", "--port", str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=log,
            )
            try:
                base = f"http://127.0.0.1:{port}"
                for _ in range(80):
                    try:
                        with opener.open(base + "/api/health", timeout=1) as response:
                            health = json.load(response)
                        break
                    except OSError:
                        if proc.poll() is not None:
                            break
                        time.sleep(0.25)
                else:
                    raise RuntimeError("Server did not become ready")
                if proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                assert health["status"] == "ok"
                sizes = {}
                for path in ["/", "/audits", "/assets/app.js", "/assets/styles.css", "/favicon.svg", "/modeltrace-mark.svg", "/logo.svg"]:
                    with opener.open(base + path, timeout=5) as response:
                        data = response.read()
                        if path.endswith(".svg"):
                            assert response.headers.get_content_type() == "image/svg+xml", path
                            assert b"<svg" in data, path
                    assert data, path
                    if path in ["/", "/audits"]:
                        assert b'<div id="root"></div>' in data
                    sizes[path] = len(data)
                assert not (ROOT / "frontend/dist/UNSUITABLE_FOR_RELEASE.json").exists()
                print(json.dumps({"status": "PASS", "health": health, "asset_bytes": sizes}, indent=2))
            finally:
                if os.name == "nt":
                    # Windows venv python.exe can launch a child interpreter; stop the entire owned tree.
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
