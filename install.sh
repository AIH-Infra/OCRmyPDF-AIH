#!/bin/bash

# OCRmyPDF-AIH Install Script (macOS/Linux)

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
echo "   OCRmyPDF-AIH Environment Setup"
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
pip install --upgrade pip -q

# ============================================
# Step 3: Install Dependencies
# ============================================
echo "[3/4] Installing OCRmyPDF-AIH..."

pip install -e . -q
echo "[OK] Dependencies installed"
echo ""

# ============================================
# Step 4: Check / Install Tesseract & Ghostscript
# ============================================
echo "[4/4] Checking Tesseract & Ghostscript..."

for prog in tesseract gs; do
    if command -v $prog &>/dev/null; then
        echo "[Found] $prog: $($prog --version 2>&1 | head -1)"
    else
        echo "[Info] $prog not found, trying auto-install..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &>/dev/null; then
                if [ "$prog" = "tesseract" ]; then
                    brew install tesseract tesseract-lang 2>/dev/null && echo "[OK] Tesseract installed" || \
                        echo "[Warning] brew install tesseract tesseract-lang"
                else
                    brew install ghostscript 2>/dev/null && echo "[OK] Ghostscript installed" || \
                        echo "[Warning] brew install ghostscript"
                fi
            else
                echo "[Warning] Homebrew not found"
            fi
        else
            if command -v apt &>/dev/null; then
                if [ "$prog" = "tesseract" ]; then
                    sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim 2>/dev/null && echo "[OK] Tesseract installed" || \
                        echo "[Warning] sudo apt install tesseract-ocr"
                else
                    sudo apt install -y ghostscript 2>/dev/null && echo "[OK] Ghostscript installed" || \
                        echo "[Warning] sudo apt install ghostscript"
                fi
            else
                echo "[Warning] Please install $prog manually"
            fi
        fi
    fi
done

echo ""
echo "============================================================"
echo "   Installation complete!"
echo "   Run ./start.sh to launch the Web UI."
echo "   URL: http://localhost:6106"
echo "============================================================"
echo ""
