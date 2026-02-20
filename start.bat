@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ==========================================
echo   OCRmyPDF-AIH Starting...
echo   Batch PDF OCR Pipeline
echo ==========================================
echo.

:: Release port 6106
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :6106 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: Check virtual environment
if not exist .venv\Scripts\activate.bat (
    echo [Error] Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Starting Web UI...
echo.
echo URL: http://localhost:6106
echo Press Ctrl+C to stop
echo.

python -m streamlit run misc\_webservice_pro.py --server.port 6106

pause
