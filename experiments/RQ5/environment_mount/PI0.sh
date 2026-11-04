#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
ENV_NAME="PI0"
REPO_DIR="${DEEPHYPERION_ROOT:-/home/ubuntu/Deephyperion_VLA}"

# 1. Initialize Conda for this shell session
# This allows 'conda activate' to work inside the script
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
echo "--- Creating Conda Environment: $ENV_NAME ---"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# Match the other model environments and retain pkg_resources during setup.
pip install "setuptools==80.10.2"


echo "--- Installing LeRobot and PI0 ---"
pip install numpy==1.24.4
echo "--- Installing ManiSkill2 real2sim ---"
cd "$REPO_DIR/ManiSkill2_real2sim"
pip install -e .

echo "--- Installing main package ---"
cd "$REPO_DIR"
pip install -e .
apt install ffmpeg

# Keep TensorFlow's Werkzeug dependency on a known compatible MarkupSafe.
pip install 'MarkupSafe==2.1.5'
if ! pip install tensorflow==2.15.0; then
    echo "TensorFlow installation failed; checking for incomplete MarkupSafe metadata." >&2
    # Repair only the specific incomplete distribution reported by pip.
    if python - <<'PYCODE'
from pathlib import Path
import shutil
import sys
import sysconfig

info = Path(sysconfig.get_paths()['purelib']) / 'markupsafe-3.0.4.dist-info'
if not info.is_dir() or info.is_symlink() or (info / 'METADATA').exists():
    sys.exit(1)
print('Removing incomplete metadata:', info, flush=True)
shutil.rmtree(info)
PYCODE
    then
        pip install --force-reinstall 'MarkupSafe==2.1.5'
        pip install tensorflow==2.15.0
    else
        echo "No matching incomplete MarkupSafe metadata found; cannot repair automatically." >&2
        exit 1
    fi
fi
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
pip install git+https://github.com/nathanrooy/simulated-annealing
pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation

echo "--- Installing transformers ---"
cd "lerobot"
pip install -e .
pip install "numpy==1.24.4"
pip install pytest
cd "$REPO_DIR/transformers-4.48.1"
pip install -e .
pip install moviepy
pip install "numpy==1.24.4"
pip install matplotlib
pip install openpyxl
pip install seaborn
pip install frechetdist
pip install matplotlib_venn

pip install scikit-learn
pip install joblib
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation flash-attn==2.7.1.post4
# Keep datasets compatible with the project's transformers and Hub versions.
pip install 'datasets==3.6.0' 'huggingface-hub==0.36.2' 'numpy==1.24.4'
echo "--- Downloading models ---"
cd "$REPO_DIR/checkpoints"
python download_model.py HaomingSong/lerobot-pi0-bridge
python download_model.py HaomingSong/lerobot-pi0-fractal

echo "--- Setup Complete! ---"
echo "To start using the environment, run: conda activate $ENV_NAME"
