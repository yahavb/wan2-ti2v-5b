# VAE on Neuron — NKI Kernel Fusion Strategy

## The Problem

The Wan2.2 VAE (CausalConv3d-based encoder/decoder) is currently running on **CPU in eager mode** because:

1. **`torch.compile` on the VAE creates 400+ NEFFs** (one per CausalConv3d op, each with its own compiled kernel)
2. **Each NEFF eats HBM** for model code + scratchpad — total ~20GB on a single NC
3. **OOM** when combined with DiT weights on the same NeuronDevice
4. The causal cache management (`feat_cache`, `feat_idx`, mutable list indexing, `torch.cat` with conditionals) creates graph breaks that prevent the Neuron compiler from fusing ops

**Result:** VAE decode takes ~5 min on CPU (the #1 bottleneck), VAE encode takes ~8 min.

## The Solution: NKI Kernel Fusion

Replace the 400+ tiny compiled ops with **~14 hand-written NKI kernels** that fuse multiple operations into single NEFFs.

### Key Insight

During decode, the VAE processes **one temporal frame at a time** (`for i in range(iter_)` in `decode()`). So each CausalConv3d with kernel `(3,3,3)` operating on `[B, C, 1, H, W]` is effectively a **spatial Conv2d** on `[B*1, C, H, W]` with temporal context from cached frames handled in eager Python. The heavy compute is 2D spatial convolution — perfect for NKI im2col-free matmul.

### NKI Kernels Implemented

#### 1. `kernels/vae_conv2d.py` — Spatial Conv2d

**`vae_conv2d_k1`**: Pointwise (1×1) convolution
- Pure matmul: `weight[C_out, C_in] @ input[C_in, HW] + bias`
- Used for: shortcut convolutions, QKV projections, output projections
- Tiled over C_out (P=128), C_in (P=128), spatial (512)

**`vae_conv2d_k3_shifted`**: 3×3 convolution via 9 shifted matmuls
- **No im2col expansion** — instead, caller pre-constructs 9 spatially-shifted input tensors
- For each kernel position (kh, kw): `weight_slice[C_out, C_in] @ shifted_input[C_in, HW]`
- Accumulate 9 matmuls + bias in float32, cast to bf16
- Caller provides `shifted_inputs: (9*C_in, HW_padded)` and `weight_slices: (C_out, C_in*9)`
- Used for: all CausalConv3d(3,3,3) in ResidualBlocks and head/tail convs

#### 2. `kernels/vae_attention.py` — VAE Self-Attention

**`vae_self_attention`**: Single-head spatial self-attention (SDPA)
- Adapted from `wan_cross_attn` kernel pattern
- Handles large d (1024 = 8×128 tiles) with tiled QK matmul accumulation
- Single-pass softmax (seq ≤ 1560, fits in SBUF)
- Identity matmul trick for transpose in PV computation
- Used for: the 2 AttentionBlocks in encoder/decoder middle blocks

### NEFF Budget After Fusion

| Component | Before (torch.compile) | After (NKI) |
|-----------|----------------------|-------------|
| ResidualBlock convs | ~160 NEFFs | 6 (one per unique shape) |
| AttentionBlocks | ~8 NEFFs | 1 |
| Resample/Upsample convs | ~12 NEFFs | 3 |
| time_conv | ~4 NEFFs | 1 |
| conv1/conv2/head | ~6 NEFFs | 3 |
| Cache management (graph breaks) | ~200+ NEFFs | 0 (stays in eager) |
| **Total** | **~400+** | **~14** |

14 NEFFs × ~50MB each ≈ **700MB model code** — easily fits alongside DiT on NC0.

### How the Caller Prepares Shifted Inputs

For a CausalConv3d with temporal frame already extracted (single-frame decode path):

```python
def prepare_shifted_conv_input(x_2d, H, W, K=3, padding=1):
    """x_2d: (C, H*W) → shifted: (9*C, HW_padded)"""
    C = x_2d.shape[0]
    x_3d = x_2d.reshape(C, H, W)
    x_padded = F.pad(x_3d, (padding, padding, padding, padding))
    shifts = []
    for kh in range(K):
        for kw in range(K):
            window = x_padded[:, kh:kh+H, kw:kw+W].reshape(C, H*W)
            shifts.append(window)
    result = torch.cat(shifts, dim=0)  # (9*C, HW)
    # Pad HW to multiple of 512
    pad_hw = (512 - (H*W) % 512) % 512
    return F.pad(result, (0, pad_hw)) if pad_hw else result
```

This runs in **eager Python on Neuron** (just tensor reshape/pad/slice — no compilation needed).

## Testing

```bash
# Run on Neuron instance:
NEURON_RT_NUM_CORES=2 python test_vae_kernels.py
```

Tests each kernel against CPU PyTorch reference with tolerance `max_diff < 0.05` (bf16).

Test configurations use production shapes from the actual VAE decoder at 480p:
- Conv2d K=1: 128→1024, 1024→512, 512→256, 1024→3072 (QKV)
- Conv2d K=3: 1024→1024 at 30×52, 60×104, 120×208
- Self-attention: d=1024, seq=1560

## Integration Plan

### Phase 1: Validate Kernels (Current)
Run `test_vae_kernels.py` — all tests must pass before proceeding.

### Phase 2: Wire into VAE
Modify `wan/modules/vae2_2.py`:
- In `ResidualBlock.forward()`: replace CausalConv3d calls with NKI `vae_conv2d_k3_shifted`
- In `AttentionBlock.forward()`: replace SDPA with NKI `vae_self_attention`
- Keep causal cache management in eager Python (no graph breaks in NKI)
- Use `wrap_nki()` to bridge NKI kernels into PyTorch

### Phase 3: Move VAE to Neuron
In `inference_neuron_ti2v.py`:
- Remove CPU fallback: `vae.model = vae.model.to(NEURON_DEVICE)`
- VAE encode/decode now use NKI kernels on Neuron
- Expected speedup: ~5 min → ~15-30s for decode

### Phase 4 (Future): TP-2 for VAE Decoder
- Column-parallel on first conv in each ResBlock (split C_out across 2 cores)
- Row-parallel on second conv (split C_in, all-reduce)
- Halves per-core activation memory
- C=1024 splits cleanly to 512/core

## Architecture Reference

### Model (Wan2.2-TI2V-5B)
- **DiT:** dim=3072, 24 heads, 30 transformer layers, ffn_dim=14336
- **VAE:** Wan2.2_VAE (CausalConv3d-based, z_dim=48, temporal stride=4, spatial stride=16)
- **T5:** UMT5-XXL encoder for text conditioning
- **VAE stride:** (4, 16, 16) — temporal, height, width
- **Patch size:** (1, 2, 2)

### TP-8 Setup
- **8 NeuronCores** (2 NDs × 4 LNCs each, `NEURON_LOGICAL_NC_CONFIG=2`)
- **DiT:** TP-sharded across all 8 ranks
- **T5:** On rank 4 (ND1:NC0), Neuron eager
- **VAE:** On rank 0 (ND0:NC0), **currently CPU → target: Neuron with NKI kernels**

### VAE Decoder Data Flow (per temporal chunk)
```
Input: [B, 48, 1, H_lat, W_lat]  (H_lat=30, W_lat=52 for 480p)

conv1: CausalConv3d(48→1024)     → NKI vae_conv2d_k3_shifted
middle:
  ResBlock(1024,1024)             → NKI vae_conv2d_k3_shifted × 2
  AttentionBlock(1024)            → NKI vae_self_attention
  ResBlock(1024,1024)             → NKI vae_conv2d_k3_shifted × 2

Up_ResBlock_0: 3×ResBlock(1024→1024) + Upsample2d    → 30×52 → 60×104
Up_ResBlock_1: 3×ResBlock(1024→1024) + Upsample3d     → 60×104 → 120×208
Up_ResBlock_2: 3×ResBlock(1024→512) + Upsample3d      → 120×208 → 240×416
Up_ResBlock_3: 3×ResBlock(512→256) (no upsample)      → 240×416

head: RMSNorm → SiLU → CausalConv3d(256→12)          → NKI vae_conv2d_k3_shifted
unpatchify: [B,12,...] → [B,3,F,H,W]
```

## Current Performance (81 frames, 480p, 20 steps)
- Model loading + TP sharding: ~10s
- T5 encoding: ~40s
- **VAE encode (CPU): ~8 min** ← target for NKI optimization
- Denoising: ~170-285s (14s/step × 20)
- **VAE decode (CPU): ~5 min** ← target for NKI optimization
- Total: ~15-20 min

## File Structure
```
wan2-ti2v-5b/
├── inference_neuron_ti2v.py       # Main inference script
├── test_vae_kernels.py            # NKI kernel accuracy tests
├── VAE_NEURON_OPTIMIZATION.md     # This file
├── setup.sh                       # Entrypoint
├── rope_neuron.py                 # Custom RoPE for 3D video
├── models/
│   └── tp_utils.py                # TP sharding utilities
├── kernels/
│   ├── cross_attention.py         # NKI cross-attention (DiT)
│   ├── self_attention.py          # NKI self-attention (DiT)
│   ├── vae_conv2d.py              # NKI Conv2d K=1 and K=3 (VAE)
│   └── vae_attention.py           # NKI self-attention (VAE)
├── wan/
│   ├── modules/
│   │   ├── attention.py           # NKI-aware attention dispatch
│   │   ├── model.py               # WanModel (DiT)
│   │   ├── vae2_2.py              # Wan2.2 VAE (with NKI dispatch)
│   │   └── ...
│   └── utils/
│       └── ...
└── wan_requirements.txt
```

## Key Decisions & Lessons
1. **Shifted-input approach** — instead of im2col (which would expand memory 9×), we pre-shift the input on the host side and pass 9 shifted views. The NKI kernel is pure matmul accumulation.
2. **No fused ResidualBlock kernel** — instead, we use vae_conv2d_k3 as a building block and keep RMSNorm + SiLU in eager. This is simpler to debug and still reduces NEFFs drastically (each unique shape → 1 NEFF instead of 6-8).
3. **Cache management stays in eager** — the causal cache logic (feat_cache, feat_idx, torch.cat) runs in Python, not in the NKI kernel. This avoids graph breaks entirely.
4. **VAE dtype fix** — `z.to(self.conv2.weight.dtype)` after scale division prevents NaN from fp32/bf16 mismatch.
