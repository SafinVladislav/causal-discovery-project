# =============================================================================
#  Causal Discovery Project – Linux/macOS Setup
# =============================================================================
error_exit() {
    echo ""
    echo "[ERROR] Something went wrong during setup." >&2
    echo "Check the error messages above." >&2
    exit 1
}

echo ""
echo "=================================================="
echo "   Causal Discovery Project - Setup"
echo "=================================================="
echo ""

if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "[ERROR] Python is not installed or not in PATH!"
    echo "Please install Python 3.9+ from your package manager or https://www.python.org/downloads/"
    exit 1
fi

PYVER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Found $PYTHON ($PYVER)"

if [ -d "venv" ]; then
    echo "Virtual environment folder exists - checking if it is healthy..."
    if [ -f "venv/bin/python" ] && venv/bin/python -m pip --version >/dev/null 2>&1; then
        echo "Existing venv looks healthy - reusing it."
    else
        echo "Existing venv is broken or incomplete → deleting and recreating it."
        rm -rf venv
    fi
fi

# ---- Create venv only if it doesn't exist ----
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON -m venv venv || error_exit
fi

# ---- Activate venv ----
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || error_exit

# ---- Upgrade pip the modern safe way ----
echo "Upgrading pip..."
python -m pip install --upgrade pip > /dev/null 2>&1 || true

# ---- Install / update requirements ----
echo ""
echo "Installing / updating project dependencies..."
pip install torch --index-url https://download.pytorch.org/whl/cpu || true
pip install -r requirements.txt --upgrade || error_exit

deactivate || true

echo ""
echo "=================================================="
echo "    SUCCESS! Everything is ready"
echo "=================================================="
echo ""
echo "To run the simulation now or later:"
echo "   source venv/bin/activate"
echo "   python run_simulation.py"
echo "   deactivate"
echo ""
echo "Tip: You can run ./setup.sh anytime - it is completely safe!"
echo ""
