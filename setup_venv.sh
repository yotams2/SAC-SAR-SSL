#!/bin/bash
# ==============================================================================
# Script to create/regenerate the SAC-SAR-SSL virtual environment on /scratch200
# Usage:
#   ./setup_venv.sh [--force] [venv_path]
# ==============================================================================

set -e

# Default venv destination on scratch storage
DEFAULT_VENV_PATH="/scratch200/yotams2/venvs/sac-sar-ssl"
FORCE_RECREATE=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --force|-f)
            FORCE_RECREATE=true
            ;;
        *)
            TARGET_VENV_PATH="$arg"
            ;;
    esac
done

VENV_PATH="${TARGET_VENV_PATH:-$DEFAULT_VENV_PATH}"

echo "=========================================================="
echo " Setting up SAC-SAR-SSL Virtual Environment"
echo " Target path: $VENV_PATH"
echo "=========================================================="

# Check if existing venv should be removed or skipped
if [ -d "$VENV_PATH" ]; then
    if [ "$FORCE_RECREATE" = true ]; then
        echo "--> [INFO] Found existing venv at $VENV_PATH. Removing (--force was passed)..."
        rm -rf "$VENV_PATH"
    else
        echo "--> [INFO] Virtual environment already exists at $VENV_PATH."
        echo "    To recreate it from scratch, re-run with: ./setup_venv.sh --force"
        exit 0
    fi
fi

# Ensure parent directory and scratch cache directory exist
mkdir -p "$(dirname "$VENV_PATH")"
mkdir -p /scratch200/yotams2/.cache/pip

# Prevent home directory quota exhaustion by redirecting pip cache to scratch
export PIP_CACHE_DIR="/scratch200/yotams2/.cache/pip"

# 1. Load cluster modules (Python 3.11 with pre-installed PyTorch + CUDA)
echo "--> [1/4] Loading cluster modules..."
module purge
module load python/python-3.11_torch-2.2.2
module load CUDA/CUDA-12.2

# Configure CUDA compiler paths for native builds
export CUDA_HOME=/powerapps/share/rocky9/CUDA/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"

# 2. Create virtual environment with system site packages enabled
# (This links to the cluster-optimized PyTorch 2.2.2 and NumPy without duplicating storage)
echo "--> [2/4] Creating virtual environment with --system-site-packages..."
python3 -m venv --system-site-packages "$VENV_PATH"

# 3. Install core repository dependencies
echo "--> [3/4] Installing Python dependencies into venv..."
"$VENV_PATH/bin/pip" install --no-cache-dir \
    scipy \
    soundfile \
    webrtcvad \
    jsonargparse \
    tqdm \
    h5py \
    mat73 \
    scikit-learn \
    tensorboardX \
    pandas

# 4. Compile and install gpuRIR from GitHub
echo "--> [4/4] Building and installing gpuRIR with CUDA support..."
"$VENV_PATH/bin/pip" install --no-cache-dir git+https://github.com/DavidDiazGuerra/gpuRIR.git

# Verify installation
echo "--> Verifying installed packages in venv..."
"$VENV_PATH/bin/python" -c "
import torch
import numpy
import soundfile
import scipy
import webrtcvad
import gpuRIR
print('Verification successful!')
print(f' - PyTorch: {torch.__version__}, NumPy: {numpy.__version__}')
"

echo "=========================================================="
echo " [SUCCESS] Virtual environment ready at:"
echo "   $VENV_PATH"
echo ""
echo " To activate it in your shell or jobs:"
echo "   module load python/python-3.11_torch-2.2.2 CUDA/CUDA-12.2"
echo "   source $VENV_PATH/bin/activate"
echo "=========================================================="
