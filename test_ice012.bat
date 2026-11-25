@echo off
REM ============================================================
REM  ICE-012 child-voice test - one click.
REM  Downloads the model on first run (~1.4 GB), then generates
REM  4 child/teen samples from the sky lesson into outputs\.
REM ============================================================
cd /d "%~dp0"

if exist venv\Scripts\python.exe (
    set PY=venv\Scripts\python.exe
) else (
    set PY=python
)

echo Checking packages...
%PY% -m pip install --quiet transformers soundfile imageio-ffmpeg accelerate
if errorlevel 1 ( echo Package install failed - send a photo of this window & pause & exit /b 1 )

%PY% test_ice012.py
pause
