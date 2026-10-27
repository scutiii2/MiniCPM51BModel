@echo off
REM Starts the MiniCPM service on port 9130 (creates .venv_minicpm on first run). Needs Ollama running with the model:
REM   ollama pull hf.co/openbmb/MiniCPM5-1B-GGUF:Q4_K_M
REM LABEL: MiniCPM Service
REM DESCRIPTION: Shared HTTP service that serves MiniCPM5-1B (via Ollama) for every project.

cd /d "%~dp0"

if not exist ".venv_minicpm\Scripts\python.exe" (
    echo Creating virtual environment .venv_minicpm ...
    py -m venv .venv_minicpm
    call .venv_minicpm\Scripts\activate
    pip install -e ".[dev]"
) else (
    call .venv_minicpm\Scripts\activate
)

python src/app.py
pause
