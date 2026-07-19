#!/bin/bash

# Define virtual environment name
VENV_DIR="qml_env"

echo "Starting environment setup..."

# Find a compatible Python interpreter (TF 2.15.0 supports 3.9 - 3.11)
PYTHON_CMD=""
for candidate in python3.9 python3.10 python3.11 python3; do
    if command -v "$candidate" &> /dev/null; then
        # Check if python3 is running a compatible minor version
        if [ "$candidate" = "python3" ]; then
            minor_ver=$(python3 -c 'import sys; print(sys.version_info.minor)')
            major_ver=$(python3 -c 'import sys; print(sys.version_info.major)')
            if [ "$major_ver" -eq 3 ] && [ "$minor_ver" -ge 9 ] && [ "$minor_ver" -le 11 ]; then
                PYTHON_CMD="python3"
                break
            fi
        else
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No compatible Python version found (3.9, 3.10, or 3.11 required for TensorFlow 2.15.0)."
    echo "Please install a compatible python version (e.g., sudo apt install python3.10 python3.10-venv) and retry."
    exit 1
else
    echo "Using compatible python interpreter: $PYTHON_CMD ($($PYTHON_CMD --version))"
fi

# Clean-up existing virtual environments
if [ -d "$VENV_DIR" ]; then
    echo "Existing virtual environment '$VENV_DIR' found. Removing it..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment in '$VENV_DIR'..."
$PYTHON_CMD -m venv "$VENV_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing required Python packages..."

pip install \
    uproot \
    pandas \
    numpy \
    sympy \
    requests \
    aiohttp \
    pyyaml \
    tensorflow==2.15.0 \
    tensorflow-quantum \
    cirq \
    scikit-learn \
    awkward \
    awkward-pandas \
    tables \
    pytest

if [ $? -ne 0 ]; then
    echo "Error: Environment installation failed."
    deactivate
    exit 1
fi

deactivate
echo -e "\nSetup Complete! Run: source $VENV_DIR/bin/activate"