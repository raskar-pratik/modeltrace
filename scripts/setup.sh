#!/usr/bin/env bash
set -euo pipefail

# Cross-platform POSIX setup script for ModelTrace (Linux / macOS)
# Strictly fail-fast: every Python, pip, npm, TypeScript (tsc --noEmit), Vite (vite build),
# and database migration step must succeed with exit code 0. Never silently converts
# dependency or typecheck failures into offline fallback success.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${MODELTRACE_VENV_DIR:-$ROOT_DIR/.venv}"

echo "[ModelTrace] Creating isolated Python virtual environment in $VENV_DIR..."
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    python3 -m venv "$VENV_DIR"
fi

echo "[ModelTrace] Installing locked Python dependencies and editable package..."
"$VENV_DIR/bin/python3" -m pip install --upgrade pip
"$VENV_DIR/bin/python3" -m pip install -r requirements.lock
"$VENV_DIR/bin/python3" -m pip install --no-deps -e .

echo "[ModelTrace] Installing and building React + TypeScript frontend via Vite..."
npm --prefix frontend ci --no-audit
npm --prefix frontend run typecheck
npm --prefix frontend run build

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[ModelTrace] Created .env from .env.example."
fi

mkdir -p data exports
"$VENV_DIR/bin/modeltrace" migrate
echo "[ModelTrace] Setup complete."
