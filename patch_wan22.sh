#!/bin/bash
# patch_wan22.sh - Patch Wan2.2 source for Neuron compatibility
# Run from the Wan2.2 repo root: bash /path/to/patch_wan22.sh
set -euo pipefail

echo "Patching Wan2.2 for Neuron..."

# t5.py: replace CUDA device with CPU
sed -i 's/device=torch.cuda.current_device()/device=torch.device("cpu")/g' wan/modules/t5.py

# model.py: remove autocast blocks
sed -i "s/with torch.amp.autocast(.*):/if True:  # autocast removed for Neuron/g" wan/modules/model.py
sed -i "s/@torch.amp.autocast(.*)/# @autocast removed for Neuron/g" wan/modules/model.py

# model.py: remove dtype asserts
sed -i '/assert e.dtype == torch.float32/d' wan/modules/model.py
sed -i '/assert e\[0\].dtype == torch.float32/d' wan/modules/model.py

# model.py: redirect flash_attention to SDPA attention
sed -i 's/from .attention import flash_attention/from .attention import attention as flash_attention/g' wan/modules/model.py

# attention.py: disable flash attention availability flags
sed -i 's/FLASH_ATTN_3_AVAILABLE = True/FLASH_ATTN_3_AVAILABLE = False/g' wan/modules/attention.py
sed -i 's/FLASH_ATTN_2_AVAILABLE = True/FLASH_ATTN_2_AVAILABLE = False/g' wan/modules/attention.py

# vae2_1.py: patch CUDA amp import
sed -i 's/import torch.cuda.amp as amp/# import torch.cuda.amp as amp (patched)/g' wan/modules/vae2_1.py

# vae2_2.py: patch CUDA amp import and usages
sed -i 's/import torch.cuda.amp as amp/import contextlib  # patched for Neuron/g' wan/modules/vae2_2.py
sed -i 's/with amp.autocast(dtype=self.dtype):/with contextlib.nullcontext():  # autocast removed for Neuron/g' wan/modules/vae2_2.py

# __init__.py: remove unused imports that pull heavy CUDA deps
sed -i 's/from .speech2video import WanS2V/# patched/g' wan/__init__.py
sed -i 's/from .animate import WanAnimate/# patched/g' wan/__init__.py
sed -i 's/from .image2video import WanI2V/# patched/g' wan/__init__.py

echo "Wan2.2 patched for Neuron successfully"
