#!/usr/bin/env python3
"""Cross-platform Python launcher for ModelTrace (`python3 scripts/launch.py`).

Responsibilities:
- Checks Python prerequisites, database migrations, and built frontend assets (`frontend/dist`).
- If `frontend/dist/index.html` is absent and project-local Vite is installed (`frontend/node_modules/.bin/vite`),
  runs `npm --prefix frontend run build`. Never silently invokes the offline fallback helper.
- Verifies localhost port availability before binding.
- Starts the single-origin ModelTrace server (serving both `/api/*` and the built UI on `http://127.0.0.1:<port>`).
- Handles SIGINT / SIGTERM cleanly and stops child processes without leaving orphan listeners.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def ensure_prerequisites(build_ui_if_missing: bool = True) -> dict:
    dist_index = ROOT_DIR / "frontend" / "dist" / "index.html"
    local_vite = (
        ROOT_DIR
        / "frontend"
        / "node_modules"
        / ".bin"
        / ("vite.cmd" if os.name == "nt" else "vite")
    )
    if build_ui_if_missing and not dist_index.is_file() and local_vite.is_file():
        subprocess.run(
            ["npm", "--prefix", "frontend", "run", "build"],
            cwd=str(ROOT_DIR),
            check=True,
        )

    venv_py = ROOT_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    python_bin = str(venv_py) if venv_py.is_file() else sys.executable

    # Run database migrations idempotently
    subprocess.run(
        [python_bin, "-m", "modeltrace.cli", "migrate"],
        cwd=str(ROOT_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
    )

    unsuitable_marker = ROOT_DIR / "frontend" / "dist" / "UNSUITABLE_FOR_RELEASE.json"
    return {
        "python_bin": python_bin,
        "frontend_dist_ready": dist_index.is_file(),
        "frontend_is_dev_fallback": unsuitable_marker.is_file(),
        "db_path": str(ROOT_DIR / "data" / "modeltrace.db"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ModelTrace Cross-Platform Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify prerequisites, migrations, UI bundle, and port availability without blocking.",
    )
    args = parser.parse_args()

    prereq = ensure_prerequisites(build_ui_if_missing=True)
    if not prereq["frontend_dist_ready"] or prereq["frontend_is_dev_fallback"]:
        print("[ModelTrace Launcher] A real production UI build is required. Run the setup script.", file=sys.stderr)
        return 1
    port_free = is_port_free(args.host, args.port)

    if args.check_only:
        print(
            f"[ModelTrace Launcher] Prerequisite check OK | UI bundle ready: {prereq['frontend_dist_ready']} "
            f"| Dev fallback marker: {prereq['frontend_is_dev_fallback']} | Port {args.host}:{args.port} free: {port_free}"
        )
        return 0 if (prereq["frontend_dist_ready"] and port_free) else 1

    if not port_free:
        print(
            f"[ModelTrace Launcher] Error: Port {args.host}:{args.port} is already in use.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        prereq["python_bin"],
        "-m",
        "modeltrace.cli",
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"[ModelTrace Launcher] Starting single-origin server on http://{args.host}:{args.port} ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR))

    def _shutdown(signum: int, _frame: object) -> None:
        print(f"\n[ModelTrace Launcher] Caught signal {signum}; stopping server cleanly...")
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
