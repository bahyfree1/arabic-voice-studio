@echo off
REM ============================================================
REM  Arabic Voice Studio - one-click installer for Windows
REM  Needs: Python 3.11 or 3.12 from python.org (tick "Add to PATH")
REM  Uses the NVIDIA GPU (GTX 1660 SUPER) via CUDA torch build.
REM ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not on PATH.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this file again.
    pause
    exit /b 1
)

python -m venv venv
if errorlevel 1 ( echo Failed to create venv & pause & exit /b 1 )
call venv\Scripts\activate.bat

python -m pip install --upgrade pip

echo Installing PyTorch with CUDA for your NVIDIA card...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo CUDA torch failed, falling back to CPU torch - generation will be slower.
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo Installing the TTS engines and the app...
pip install habibi-tts gradio mishkal soundfile transformers torchaudio pydub pyworld
if errorlevel 1 ( echo Install failed - send a photo of this window & pause & exit /b 1 )

echo.
echo ============================================================
echo  Done. From now on just double-click run.bat
echo  First generation downloads the models (about 2-3 GB for Habibi,
echo  plus 1.4 GB the first time you use the ICE-012 child voices).
echo ============================================================
pause
