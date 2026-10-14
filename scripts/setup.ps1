# Cross-platform PowerShell setup script for ModelTrace (Windows)
# Strictly fail-fast: checks $LASTEXITCODE after every native process (python, pip, npm, modeltrace)
# and stops immediately on any error without falling back to an offline bundle or printing Setup complete.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ModelTrace] FATAL: Step '$StepName' failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

Write-Host "[ModelTrace] Creating isolated Python virtual environment in .venv..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-NativeStep "python -m venv .venv" { python -m venv .venv }
}

Write-Host "[ModelTrace] Installing locked Python dependencies and editable package..."
Invoke-NativeStep "pip upgrade" { .\.venv\Scripts\python.exe -m pip install --upgrade pip }
Invoke-NativeStep "pip install requirements.lock" { .\.venv\Scripts\python.exe -m pip install -r requirements.lock }
Invoke-NativeStep "pip install editable modeltrace" { .\.venv\Scripts\python.exe -m pip install --no-deps -e . }

Write-Host "[ModelTrace] Installing and building React + TypeScript frontend via Vite..."
Invoke-NativeStep "npm ci" { npm --prefix frontend ci --no-audit }
Invoke-NativeStep "npm run typecheck (tsc --noEmit)" { npm --prefix frontend run typecheck }
Invoke-NativeStep "npm run build (vite build)" { npm --prefix frontend run build }

if ((-not (Test-Path ".env")) -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[ModelTrace] Created .env from .env.example."
}

New-Item -ItemType Directory -Force -Path "data", "exports" | Out-Null
Invoke-NativeStep "modeltrace migrate" { .\.venv\Scripts\modeltrace.exe migrate }

Write-Host "[ModelTrace] Setup complete. Start server with: .\.venv\Scripts\python.exe .\scripts\launch.py"
