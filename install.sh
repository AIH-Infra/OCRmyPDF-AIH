#!/bin/bash

# AIH-OCR Install Script (macOS/Linux)

set -e
cd "$(dirname "$0")"

# Ensure Homebrew paths are available (macOS)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Auto-install Homebrew on macOS if missing
if [[ "$OSTYPE" == "darwin"* ]] && ! command -v brew &>/dev/null; then
    echo "[Info] Homebrew not found, installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
fi

echo ""
echo "============================================================"
echo "   AIH-OCR Environment Setup"
echo "   Batch PDF OCR Pipeline"
echo "============================================================"
echo ""

# ============================================
# Step 1: Check Python
# ============================================
echo "[1/4] Checking Python..."

PYTHON_CMD=""
for ver in python3.13 python3.12 python3.11; do
    if command -v $ver &> /dev/null; then
        PYTHON_CMD=$ver
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            PYMINOR=$($cmd -c "import sys; print(sys.version_info.minor)")
            if [ "$PYMINOR" -ge 11 ] && [ "$PYMINOR" -le 13 ] 2>/dev/null; then
                PYTHON_CMD=$cmd
                break
            fi
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "[Error] Python 3.11-3.13 not found"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install python@3.12"
    else
        echo "  sudo apt install python3.12 python3.12-venv"
    fi
    exit 1
fi

PYVER=$($PYTHON_CMD --version 2>&1)
echo "[OK] $PYVER"
echo ""

# ============================================
# Step 2: Create Virtual Environment
# ============================================
echo "[2/4] Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "[Info] Found existing .venv"
    read -p "Recreate? (y/N): " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        rm -rf .venv
        $PYTHON_CMD -m venv .venv
    fi
else
    $PYTHON_CMD -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "[Error] Failed to create virtual environment"
    exit 1
fi

echo "[OK] Virtual environment ready"
echo ""

source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip -q

# ============================================
# Step 3: Install Dependencies
# ============================================
echo "[3/4] Installing AIH-OCR..."

python -m pip install -e . -c requirements-lock.txt -q
echo "[OK] Dependencies installed"
echo ""

# ============================================
# Step 4: Check / Install Tesseract & Ghostscript
# ============================================
echo "[4/4] Checking Tesseract & Ghostscript..."

COMMON_TESSERACT_LANGS="eng chi_sim chi_tra deu fra jpn kor lat"

refresh_shell_commands() {
    hash -r 2>/dev/null || true
}

ensure_brew_in_path() {
    if command -v brew &>/dev/null; then
        eval "$(brew shellenv)" 2>/dev/null || true
    fi
}

detect_tesseract() {
    if command -v tesseract &>/dev/null; then
        return 0
    fi

    for candidate in \
        /opt/homebrew/bin/tesseract \
        /usr/local/bin/tesseract \
        /usr/bin/tesseract
    do
        if [ -x "$candidate" ]; then
            export PATH="$(dirname "$candidate"):$PATH"
            refresh_shell_commands
            return 0
        fi
    done

    return 1
}

detect_ghostscript() {
    if command -v gs &>/dev/null; then
        return 0
    fi

    for candidate in \
        /opt/homebrew/bin/gs \
        /usr/local/bin/gs \
        /usr/bin/gs
    do
        if [ -x "$candidate" ]; then
            export PATH="$(dirname "$candidate"):$PATH"
            refresh_shell_commands
            return 0
        fi
    done

    return 1
}

report_tesseract_languages() {
    if ! detect_tesseract; then
        return 0
    fi

    available_langs="$(tesseract --list-langs 2>/dev/null | tail -n +2 | tr '\n' ' ')"
    missing_langs=""
    for lang in $COMMON_TESSERACT_LANGS; do
        case " $available_langs " in
            *" $lang "*) ;;
            *) missing_langs="$missing_langs $lang" ;;
        esac
    done

    if [ -z "$missing_langs" ]; then
        echo "[OK] Common Tesseract language packs available: $COMMON_TESSERACT_LANGS"
    else
        echo "[Warning] Missing common Tesseract language packs:$missing_langs"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "          Try: brew install tesseract-lang"
        elif command -v apt &>/dev/null; then
            echo "          Try: sudo apt install tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-jpn tesseract-ocr-kor tesseract-ocr-lat"
        else
            echo "          Install the missing .traineddata files for your Tesseract installation."
        fi
    fi
}

if detect_tesseract; then
    echo "[Found] tesseract: $(tesseract --version 2>&1 | head -1)"
else
    echo "[Info] tesseract not found, trying auto-install..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            ensure_brew_in_path
            if brew install tesseract 2>/dev/null; then
                brew install tesseract-lang 2>/dev/null || \
                    echo "[Info] Common language package tesseract-lang was not installed"
            else
                echo "[Warning] brew install tesseract failed"
            fi
            ensure_brew_in_path
            refresh_shell_commands
        else
            echo "[Warning] Homebrew not found"
        fi
    else
        if command -v apt &>/dev/null; then
            sudo apt update 2>/dev/null || true
            if sudo apt install -y tesseract-ocr 2>/dev/null; then
                sudo apt install -y \
                    tesseract-ocr-eng \
                    tesseract-ocr-chi-sim \
                    tesseract-ocr-chi-tra \
                    tesseract-ocr-deu \
                    tesseract-ocr-fra \
                    tesseract-ocr-jpn \
                    tesseract-ocr-kor \
                    tesseract-ocr-lat 2>/dev/null || \
                    echo "[Info] Some common Tesseract language packs were not installed"
            else
                echo "[Warning] sudo apt install tesseract-ocr failed"
            fi
            refresh_shell_commands
        else
            echo "[Warning] Please install tesseract manually"
        fi
    fi

    if detect_tesseract; then
        echo "[OK] tesseract is now available: $(tesseract --version 2>&1 | head -1)"
    else
        echo "[Warning] Tesseract install did not become available in this shell."
        echo "          Please install it manually or open a new terminal and try again."
    fi
fi

report_tesseract_languages

if detect_ghostscript; then
    echo "[Found] gs: $(gs --version 2>&1 | head -1)"
else
    echo "[Info] gs not found, trying auto-install..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            ensure_brew_in_path
            brew install ghostscript 2>/dev/null && echo "[OK] Ghostscript installed" || \
                echo "[Warning] brew install ghostscript"
            ensure_brew_in_path
            refresh_shell_commands
        else
            echo "[Warning] Homebrew not found"
        fi
    else
        if command -v apt &>/dev/null; then
            sudo apt update 2>/dev/null || true
            sudo apt install -y ghostscript 2>/dev/null && echo "[OK] Ghostscript installed" || \
                echo "[Warning] sudo apt install ghostscript"
            refresh_shell_commands
        else
            echo "[Warning] Please install gs manually"
        fi
    fi

    if detect_ghostscript; then
        echo "[OK] gs is now available: $(gs --version 2>&1 | head -1)"
    else
        echo "[Warning] Ghostscript install did not become available in this shell."
    fi
fi

# Optional: PaddleOCR. Not installed by default (Chrome ScreenAI is the main engine).
# The helper detects the GPU / CUDA driver, recommends a CPU or GPU build, asks first,
# and installs into a separate .venv-paddle. Set AIH_INSTALL_PADDLE=auto|cpu|gpu|skip
# to answer without a prompt.
python scripts/install_paddle.py || \
    echo "[Warning] Optional PaddleOCR step did not complete. Core install is fine."

echo ""
echo "============================================================"
echo "   Installation complete!"
echo "   Run ./start.sh to launch the Web UI."
echo "   The app starts from port 8501 and will use the next"
echo "   available port if needed, then open in your default browser."
echo "============================================================"
echo ""
