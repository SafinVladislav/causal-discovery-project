#!/bin/bash
# =============================================================================
#  Causal Discovery Project – One-click Linux/macOS Setup
#  Just run: ./setup.sh   (or double-click in most desktop environments)
# =============================================================================

set -e  # Stop on any error (makes the script much safer)

echo
echo "=================================================="
echo "   Causal Discovery Project - One-click Setup"
echo "=================================================="
echo

# ---- 1. Find python (python3 or python) ----
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "[ERROR] Python is not installed or not in PATH!"
    echo "Please install Python 3.9+ from your package manager or https://www.python.org/downloads/"
    exit 1
fi

# Show version
PYVER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Found $PYTHON ($PYVER)"

# ---- 2. Create venv only if it doesn't exist ----
if [ ! -d "venv" ]; then
    echo
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo
    echo "Virtual environment already exists - reusing it."
fi

# ---- 3. Activate venv ----
echo
echo "Activating virtual environment..."
source venv/bin/activate

# ---- 4. Upgrade pip the modern safe way ----
echo "Upgrading pip..."
python -m pip install --upgrade pip > /dev/null 2>&1 || true

# ---- 5. Install / update requirements ----
echo
echo "Installing / updating project dependencies..."
pip install -r requirements.txt --upgrade

deactivate || true

echo
echo "=================================================="
echo "    SUCCESS! Everything is ready"
echo "=================================================="
echo
echo "To run the simulation now or later:"
echo "   source venv/bin/activate"
echo "   python run_simulation.py"
echo "   deactivate"
echo
echo "Tip: You can run ./setup.sh anytime - it's completely safe!"
echo

# Keep terminal open if double-clicked (common on desktop environments)
if [[ -t 1 ]]; then
    exec bash  # keeps the activated venv open
else
    read -p "Press Enter to close..."
fi