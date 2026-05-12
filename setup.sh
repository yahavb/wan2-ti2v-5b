#!/bin/bash
# setup.sh - Main entrypoint for Wan2.2-TI2V-5B on Neuron
# Clones deps, installs packages, patches Wan2.2, downloads model, launches torchrun
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Clone repos ─────────────────────────────────────────────
cd /tmp
git clone https://github.com/Wan-Video/Wan2.2.git
git clone -b rolling-forcing https://yahavb:${GITHUB_TOKEN}@github.com/aws-neuron/aws-neuron-eks-samples.git

# ─── Install deps from Wan2.2 ────────────────────────────────
cd /tmp/Wan2.2
sed -i '/flash_attn/d' requirements.txt
sed -i '/torchaudio/d' requirements.txt
uv pip install -r requirements.txt
uv pip install "setuptools<81"
uv pip install git+https://github.com/pytorch/vision.git@v0.25.0 --no-deps --no-cache --no-build-isolation

# ─── Patch Wan2.2 for Neuron ─────────────────────────────────
cd /tmp/Wan2.2
bash "${SCRIPT_DIR}/patch_wan22.sh"

# ─── Model caching (S3-backed PVC tar pattern) ───────────────
MODEL_TAR="/var/mdl/wan2_2_ti2v/Wan2.2-TI2V-5B.tar"
MODEL_LOCAL="/tmp/Wan2.2-TI2V-5B"

if [[ -f "$MODEL_TAR" ]]; then
  echo "Copying model tar from S3 cache..."
  cp "$MODEL_TAR" /tmp/Wan2.2-TI2V-5B.tar
  echo "Extracting..."
  tar xf /tmp/Wan2.2-TI2V-5B.tar -C /tmp/
  rm -f /tmp/Wan2.2-TI2V-5B.tar
else
  echo "Downloading model from HuggingFace..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir='${MODEL_LOCAL}', local_dir_use_symlinks=False)
"
  echo "Creating tar archive for S3 cache..."
  tar cf /tmp/Wan2.2-TI2V-5B.tar -C /tmp Wan2.2-TI2V-5B
  mkdir -p "$(dirname $MODEL_TAR)"
  cp /tmp/Wan2.2-TI2V-5B.tar "$MODEL_TAR"
  rm -f /tmp/Wan2.2-TI2V-5B.tar
  echo "Cached tar to S3!"
fi
echo "Model weights ready at $MODEL_LOCAL"

# ─── Launch with torchrun (TP=4) ─────────────────────────────
export WAN_DIR=/tmp/Wan2.2
export RF_DIR=/tmp/aws-neuron-eks-samples/rolling-forcing/app
export MODEL_PATH=/tmp/Wan2.2-TI2V-5B

cd /tmp/Wan2.2
torchrun --nproc_per_node=${TP_DEGREE:-4} --master_port=29500 \
  "${SCRIPT_DIR}/inference_neuron_ti2v.py" 2>&1

echo "═══════════════════════════════════════════"
echo "  Done!"
echo "═══════════════════════════════════════════"
