@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo    OCRmyPDF-AIH Environment Setup
echo    Batch PDF OCR Pipeline
echo ============================================================
echo.

:: Step 1: Check Python
echo [1/4] Checking Python...

set PYTHON_CMD=

where py >nul 2>&1
if %errorlevel%==0 (
    for %%V in (3.13 3.12 3.11) do (
        if "!PYTHON_CMD!"=="" (
            py -%%V --version >nul 2>&1
            if !errorlevel!==0 (
                set PYTHON_CMD=py -%%V
                for /f "tokens=2" %%v in ('py -%%V --version 2^>^&1') do echo [Found] Python %%v
            )
        )
    )
)

if "!PYTHON_CMD!"=="" (
    where python >nul 2>&1
    if !errorlevel!==0 (
        for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
        for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
            set PYMAJOR=%%a
            set PYMINOR=%%b
        )
        if !PYMAJOR!==3 if !PYMINOR! geq 11 if !PYMINOR! leq 13 (
            set PYTHON_CMD=python
            echo [Found] Python !PYVER!
        )
    )
)

if "!PYTHON_CMD!"=="" (
    echo.
    echo [Error] No compatible Python found
    echo Required: Python 3.11, 3.12, or 3.13
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo.

:: Step 2: Create Virtual Environment
echo [2/4] Creating virtual environment...

if exist .venv (
    echo [Info] Found existing .venv
    set /p RECREATE="Recreate? (y/N): "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q .venv
        !PYTHON_CMD! -m venv .venv
    )
) else (
    !PYTHON_CMD! -m venv .venv
)

if not exist .venv\Scripts\activate.bat (
    echo [Error] Failed to create virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment ready
echo.

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q

:: Step 3: Install Dependencies
echo [3/4] Installing OCRmyPDF-AIH...
echo.

pip install -e . -q
if %errorlevel% neq 0 (
    echo [Error] Installation failed
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

:: Step 4: Check Tesseract & Ghostscript
echo [4/4] Checking Tesseract ^& Ghostscript...

:: -- Tesseract --
set TESS_OK=0
where tesseract >nul 2>&1 && set TESS_OK=1
if "!TESS_OK!"=="0" (
    if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set TESS_OK=1
    if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" set TESS_OK=1
)

if "!TESS_OK!"=="1" (
    for /f "tokens=*" %%T in ('tesseract --version 2^>^&1 ^| findstr /r "^tesseract"') do (
        echo [Found] %%T
    )
) else (
    echo [Info] Tesseract not found, trying auto-install...
    winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Tesseract installed. Reopen terminal for PATH update.
    ) else (
        echo [Warning] Auto-install failed. Install manually:
        echo           https://github.com/UB-Mannheim/tesseract/wiki
    )
)

:: -- Ghostscript --
set GS_OK=0
where gswin64c >nul 2>&1 && set GS_OK=1
if "!GS_OK!"=="0" (
    where gs >nul 2>&1 && set GS_OK=1
)
if "!GS_OK!"=="0" (
    if exist "C:\Program Files\gs\gs10.04.0\bin\gswin64c.exe" set GS_OK=1
    if exist "C:\Program Files\gs\gs10.03.1\bin\gswin64c.exe" set GS_OK=1
)

if "!GS_OK!"=="1" (
    echo [Found] Ghostscript
) else (
    echo [Info] Ghostscript not found, trying auto-install...
    winget install -e --id ArtifexSoftware.GhostScript.Console --accept-source-agreements --accept-package-agreements >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Ghostscript installed. Reopen terminal for PATH update.
    ) else (
        echo [Warning] Auto-install failed. Install manually:
        echo           https://ghostscript.com/releases/gsdnld.html
    )
)

echo.
echo ============================================================
echo    Installation complete!
echo    Run start.bat to launch the Web UI.
echo    URL: http://localhost:6106
echo ============================================================
echo.

pause
