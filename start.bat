@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ==========================================
echo   OCRmyPDF-AIH Starting...
echo ==========================================
echo.

if not exist .venv\Scripts\activate.bat (
    echo [Error] Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHONWARNINGS=ignore
if exist "%CD%\.tessdata" set "TESSDATA_PREFIX=%CD%\.tessdata"

echo Web UI will choose an available port starting from 8501.
echo Your default browser should open automatically.
echo If 8501 is unavailable or reserved by Windows, the app will try the next port.
ocrmypdf-aih-webui

pause
