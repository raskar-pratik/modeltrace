# Start ModelTrace

Extract the ZIP, then open a terminal inside the ModelTrace-1.0.0 folder.

You need Python 3.11 or 3.12 and Node.js 20 or newer. Initial setup downloads dependencies. No API key is needed for the synthetic demo.

**Linux / macOS**

```bash
bash scripts/setup.sh
.venv/bin/python3 scripts/launch.py
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe .\scripts\launch.py
```

Open http://127.0.0.1:8000 → New Audit → Quick Demo: Substituted. The genuine and mixed demos are available beside it. DEMO results are synthetic, not real LLM accuracy measurements.

Stop the server with Ctrl+C. Read README.md for real-endpoint setup, optional checkpoint support, and test commands. Read VALIDATION_REPORT.md before treating this release as a verified deployment. The GitHub release workflow runs browser, Docker, native Windows, and local HuggingFace checks. See the validation report for the dated results.
