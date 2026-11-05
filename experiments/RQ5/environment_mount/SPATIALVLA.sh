#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
ENV_NAME="SPATIALVLA"
REPO_DIR="${DEEPHYPERION_ROOT:-/home/ubuntu/Deephyperion_VLA}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

# 1. Initialize Conda for this shell session
# This allows 'conda activate' to work inside the script
# 1. Initialize Conda for this shell session
# This allows 'conda activate' to work inside the script
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
if [[ "${SPATIALVLA_RESUME:-0}" != 1 ]]; then
    echo "--- Creating Conda Environment: $ENV_NAME ---"
    conda create -n "$ENV_NAME" python=3.10 -y
else
    echo "--- Reusing partially installed environment: $ENV_NAME ---"
fi
conda activate "$ENV_NAME"




if [[ "${SPATIALVLA_RESUME:-0}" != 1 ]]; then
echo "--- Installing base dependencies ---"
pip install numpy==1.24.4
echo "--- Installing ManiSkill2 real2sim ---"
cd "$REPO_DIR/ManiSkill2_real2sim"
pip install -e .

echo "--- Installing main package ---"
cd "$REPO_DIR"
pip install -e .
apt install ffmpeg

pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
fi

# Install the required PyTorch first: flash-attn imports torch while building.
# Match the installed CUDA 12.1 toolkit and avoid pip's isolated build env.
cd "$REPO_DIR"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ninja
# Verbose pip forwards Ninja output; NINJA_STATUS shows completed/total jobs.
echo "--- Building SpatialVLA requirements (MAX_JOBS=${MAX_JOBS:-8}) ---"
MAX_JOBS="${MAX_JOBS:-8}" NINJA_STATUS='[%f/%t %p] ' \
    pip install -v --no-build-isolation -r spatialVLA_requirements.txt
echo "--- Installing transformers ---"
cd "transformers-4.48.1"
pip install -e .
pip install moviepy
pip install "numpy==1.24.4"
pip install matplotlib
pip install openpyxl
pip install seaborn
pip install frechetdist
pip install matplotlib_venn
pip install "setuptools==80.10.2"
pip install joblib
pip install scikit-learn
echo "--- Downloading models ---"
cd "$REPO_DIR/checkpoints"
python download_model.py IPEC-COMMUNITY/spatialvla-4b-mix-224-pt
cp modeling_spatialvla.py spatialvla-4b-mix-224-pt

echo "--- Setup Complete! ---"
echo "To start using the environment, run: conda activate $ENV_NAME"
