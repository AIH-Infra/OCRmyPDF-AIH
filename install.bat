@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo    AIH-OCR Environment Setup
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
python -m ensurepip --upgrade
if errorlevel 1 exit /b 1
python -m pip install --upgrade pip -q

:: Step 3: Install Dependencies
echo [3/4] Installing AIH-OCR...
echo.

python -m pip install -e . -c requirements-lock.txt -q
if %errorlevel% neq 0 (
    echo [Error] Installation failed
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

:: Step 4: Check Tesseract & Ghostscript
echo [4/4] Checking Tesseract ^& Ghostscript...

set "COMMON_TESSERACT_LANGS=eng chi_sim chi_tra deu fra jpn kor lat"
set "PROJECT_TESSDATA=%CD%\.tessdata"
set "TESS_EXE="
set "GS_EXE="

call :ensure_tesseract
call :install_common_tesseract_languages
call :seed_tessdata_support
call :report_tesseract_languages
call :ensure_ghostscript

goto :after_dependency_checks

:ensure_tesseract
call :detect_tesseract
if defined TESS_EXE (
    call :print_tesseract_version
    goto :eof
)

echo [Info] Tesseract not found, trying auto-install...
where winget >nul 2>&1
if errorlevel 1 (
    echo [Warning] winget not found. Install Tesseract manually:
    echo           https://github.com/UB-Mannheim/tesseract/wiki
    goto :eof
)

winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements >nul 2>&1
if errorlevel 1 (
    echo [Warning] Auto-install failed. Install manually:
    echo           https://github.com/UB-Mannheim/tesseract/wiki
    goto :eof
)

call :detect_tesseract
if defined TESS_EXE (
    echo [OK] Tesseract is now available: !TESS_EXE!
) else (
    echo [Warning] Tesseract installer completed, but tesseract.exe is not available in this shell.
    echo           Open a new terminal or add the install directory to PATH.
)
goto :eof

:ensure_ghostscript
call :detect_ghostscript
if defined GS_EXE (
    echo [Found] Ghostscript at !GS_EXE!
    goto :eof
)

echo [Info] Ghostscript not found, trying auto-install...
where winget >nul 2>&1
if errorlevel 1 (
    echo [Warning] winget not found. Install Ghostscript manually:
    echo           https://ghostscript.com/releases/gsdnld.html
    goto :eof
)

winget install -e --id ArtifexSoftware.GhostScript.Console --accept-source-agreements --accept-package-agreements >nul 2>&1
if errorlevel 1 (
    echo [Warning] Auto-install failed. Install manually:
    echo           https://ghostscript.com/releases/gsdnld.html
    goto :eof
)

call :detect_ghostscript
if defined GS_EXE (
    echo [OK] Ghostscript is now available: !GS_EXE!
) else (
    echo [Warning] Ghostscript installer completed, but gs is not available in this shell.
    echo           Open a new terminal or add the install directory to PATH.
)
goto :eof

:print_tesseract_version
set "TESS_VERSION_FILE=%TEMP%\ocrmypdf_aih_tesseract_version.txt"
"!TESS_EXE!" --version > "!TESS_VERSION_FILE!" 2>nul
for /f "tokens=*" %%T in ('findstr /r "^tesseract" "!TESS_VERSION_FILE!" 2^>nul') do (
    echo [Found] %%T
    del "!TESS_VERSION_FILE!" >nul 2>&1
    goto :eof
)
del "!TESS_VERSION_FILE!" >nul 2>&1
echo [Found] Tesseract at !TESS_EXE!
goto :eof

:detect_tesseract
set "TESS_EXE="
for %%P in (tesseract.exe) do (
    for /f "delims=" %%I in ('where %%P 2^>nul') do (
        set "TESS_EXE=%%~fI"
        goto :eof
    )
)
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    set "TESS_EXE=C:\Program Files\Tesseract-OCR\tesseract.exe"
    set "PATH=C:\Program Files\Tesseract-OCR;!PATH!"
    goto :eof
)
if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" (
    set "TESS_EXE=C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    set "PATH=C:\Program Files (x86)\Tesseract-OCR;!PATH!"
    goto :eof
)
goto :eof

:collect_tesseract_languages
set "MISSING_LANGS="
set "LANG_LIST="
if not defined TESS_EXE goto :eof
set "TESS_LANG_FILE=%TEMP%\ocrmypdf_aih_tesseract_langs.txt"
set "TESSDATA_PREFIX=%PROJECT_TESSDATA%"
"!TESS_EXE!" --list-langs > "!TESS_LANG_FILE!" 2>nul
for /f "delims=" %%L in ('type "!TESS_LANG_FILE!" 2^>nul') do (
    if not "%%L"=="List of available languages" set "LANG_LIST=!LANG_LIST! %%L"
)
del "!TESS_LANG_FILE!" >nul 2>&1
for %%L in (!COMMON_TESSERACT_LANGS!) do (
    echo !LANG_LIST! | findstr /c:" %%L" >nul 2>&1
    if !errorlevel! neq 0 set "MISSING_LANGS=!MISSING_LANGS! %%L"
)
goto :eof

:install_common_tesseract_languages
if not defined TESS_EXE goto :eof
if not exist "!PROJECT_TESSDATA!" mkdir "!PROJECT_TESSDATA!"
call :collect_tesseract_languages
if "!MISSING_LANGS!"=="" goto :eof
where curl >nul 2>&1
if errorlevel 1 (
    echo [Warning] curl not found. Missing common Tesseract language packs:!MISSING_LANGS!
    echo           Install missing .traineddata files into !PROJECT_TESSDATA!
    goto :eof
)
echo [Info] Downloading missing common Tesseract language packs to .tessdata...
for %%L in (!MISSING_LANGS!) do (
    if not exist "!PROJECT_TESSDATA!\%%L.traineddata" (
        curl -L --fail --silent --show-error -o "!PROJECT_TESSDATA!\%%L.traineddata" "https://github.com/tesseract-ocr/tessdata_fast/raw/main/%%L.traineddata"
        if errorlevel 1 (
            del "!PROJECT_TESSDATA!\%%L.traineddata" >nul 2>&1
            echo [Warning] Failed to download %%L.traineddata
        ) else (
            echo [OK] Downloaded %%L.traineddata
        )
    )
)
goto :eof

:seed_tessdata_support
:: .tessdata is used as TESSDATA_PREFIX by start.bat. Tesseract also looks there for its
:: configs\ folder (hocr, txt, pdf...) and osd.traineddata, so copy them from the system
:: installation. Without them OCR fails with "read_params_file: Can't open hocr".
if not defined TESS_EXE goto :eof
for %%I in ("!TESS_EXE!") do set "TESS_HOME=%%~dpI"
set "SYS_TESSDATA=!TESS_HOME!tessdata"
if not exist "!SYS_TESSDATA!\configs" goto :eof
if not exist "!PROJECT_TESSDATA!\configs" xcopy /e /i /y "!SYS_TESSDATA!\configs" "!PROJECT_TESSDATA!\configs" >nul
if not exist "!PROJECT_TESSDATA!\tessconfigs" if exist "!SYS_TESSDATA!\tessconfigs" xcopy /e /i /y "!SYS_TESSDATA!\tessconfigs" "!PROJECT_TESSDATA!\tessconfigs" >nul
if not exist "!PROJECT_TESSDATA!\osd.traineddata" if exist "!SYS_TESSDATA!\osd.traineddata" copy /y "!SYS_TESSDATA!\osd.traineddata" "!PROJECT_TESSDATA!\osd.traineddata" >nul
if not exist "!PROJECT_TESSDATA!\pdf.ttf" if exist "!SYS_TESSDATA!\pdf.ttf" copy /y "!SYS_TESSDATA!\pdf.ttf" "!PROJECT_TESSDATA!\pdf.ttf" >nul
goto :eof

:report_tesseract_languages
if not defined TESS_EXE goto :eof
call :collect_tesseract_languages
if "!MISSING_LANGS!"=="" (
    echo [OK] Common Tesseract language packs available: !COMMON_TESSERACT_LANGS!
) else (
    echo [Warning] Missing common Tesseract language packs:!MISSING_LANGS!
    echo           The Windows installer may not include every language pack.
    echo           Install missing .traineddata files into !PROJECT_TESSDATA! if needed.
)
goto :eof

:detect_ghostscript
set "GS_EXE="
for %%P in (gswin64c.exe gswin32c.exe gs.exe) do (
    for /f "delims=" %%I in ('where %%P 2^>nul') do (
        set "GS_EXE=%%~fI"
        goto :eof
    )
)
for /d %%D in ("C:\Program Files\gs\gs*" "C:\Program Files (x86)\gs\gs*") do (
    if exist "%%~fD\bin\gswin64c.exe" (
        set "GS_EXE=%%~fD\bin\gswin64c.exe"
        set "PATH=%%~fD\bin;!PATH!"
        goto :eof
    )
    if exist "%%~fD\bin\gswin32c.exe" (
        set "GS_EXE=%%~fD\bin\gswin32c.exe"
        set "PATH=%%~fD\bin;!PATH!"
        goto :eof
    )
)
for /f "tokens=2,*" %%A in ('reg query "HKLM\SOFTWARE\GPL Ghostscript" /s /v GS_DLL 2^>nul ^| findstr /i "GS_DLL"') do (
    set "GS_DLL=%%B"
    for %%G in ("!GS_DLL!") do (
        if exist "%%~dpGgswin64c.exe" (
            set "GS_EXE=%%~dpGgswin64c.exe"
            set "PATH=%%~dpG;!PATH!"
            goto :eof
        )
        if exist "%%~dpGgswin32c.exe" (
            set "GS_EXE=%%~dpGgswin32c.exe"
            set "PATH=%%~dpG;!PATH!"
            goto :eof
        )
    )
)
goto :eof

:after_dependency_checks

:: Optional: PaddleOCR. Not installed by default (Chrome ScreenAI is the main engine).
:: The helper detects the GPU / CUDA driver, recommends a CPU or GPU build, asks first,
:: and installs into a separate .venv-paddle. Set AIH_INSTALL_PADDLE=auto|cpu|gpu|skip
:: to answer without a prompt.
python scripts\install_paddle.py
if errorlevel 1 echo [Warning] Optional PaddleOCR step did not complete. Core install is fine.

echo.
echo ============================================================
echo    Installation complete!
echo    Run start.bat to launch the Web UI.
echo    The app will start from port 8501 and automatically use the next
echo    available port if needed, then open in your default browser.
echo ============================================================
echo.

pause
