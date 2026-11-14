@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ==========================================
echo   AIH-OCRmyPDF Starting...
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
set PYTHONDONTWRITEBYTECODE=1
:: Use the project .tessdata only when it is complete (configs\ present); repair it from
:: the system Tesseract if an older install.bat left it without configs\.
if exist "%CD%\.tessdata" if not exist "%CD%\.tessdata\configs" (
    for %%D in ("C:\Program Files\Tesseract-OCR\tessdata" "C:\Program Files (x86)\Tesseract-OCR\tessdata") do (
        if exist "%%~D\configs" if not exist "%CD%\.tessdata\configs" xcopy /e /i /y "%%~D\configs" "%CD%\.tessdata\configs" >nul
    )
)
if exist "%CD%\.tessdata\configs" set "TESSDATA_PREFIX=%CD%\.tessdata"

echo Web UI will choose an available port starting from 8501.
echo Your default browser should open automatically.
echo If 8501 is unavailable or reserved by Windows, the app will try the next port.
python -m ocrmypdf.webui_launcher

pause
