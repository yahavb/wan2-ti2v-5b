#!/bin/bash
# setup.sh - Main entrypoint for Wan2.2-TI2V-5B on Neuron
# Installs deps, downloads model, launches torchrun
# Wan2.2 source is pre-patched and included in this repo (wan/ directory)
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Install deps ─────────────────────────────────────────────
cd "${SCRIPT_DIR}"
# Remove CUDA-only deps from requirements
sed -i '/flash_attn/d' wan_requirements.txt
sed -i '/torchaudio/d' wan_requirements.txt
uv pip install -r wan_requirements.txt
uv pip install "setuptools<81"
uv pip install git+https://github.com/pytorch/vision.git@v0.25.0 --no-deps --no-cache --no-build-isolation
uv pip install imageio imageio-ffmpeg

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

# ─── Launch with torchrun (TP=8) ─────────────────────────────
export WAN_DIR="${SCRIPT_DIR}"
export MODEL_PATH=/tmp/Wan2.2-TI2V-5B

cd "${SCRIPT_DIR}"
torchrun --nproc_per_node=${TP_DEGREE:-8} --master_port=29500 \
  "${SCRIPT_DIR}/inference_neuron_ti2v.py" 2>&1 || true

# Copy output video to S3-backed PVC (shutil.copy fails on S3 FUSE)
if [[ -f /tmp/wan2_ti2v_output.mp4 ]]; then
  mkdir -p /var/mdl/wan2_2_ti2v/outputs
  cp /tmp/wan2_ti2v_output.mp4 /var/mdl/wan2_2_ti2v/outputs/
  echo "Video copied to /var/mdl/wan2_2_ti2v/outputs/"
fi

echo "═══════════════════════════════════════════"
echo "  Done!"
echo "═══════════════════════════════════════════"
