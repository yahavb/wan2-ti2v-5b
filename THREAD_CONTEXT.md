# Wan2.2-TI2V-5B on PyTorch Native (Neuron) — Thread Context

## Project Overview
Running **Wan-AI/Wan2.2-TI2V-5B** (text/image-to-video diffusion model) on **AWS Trainium2** (trn2.48xlarge) using **PyTorch Native eager mode** with `torch.compile(backend='neuron')` for hot sub-modules.

- **Repo:** `github.com/yahavb/wan2-ti2v-5b`
- **DLC Image:** `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-d1c940d:latest`
- **K8s Job:** `~/k8s/clusters/ray/wan2-ti2v-5b-job.yaml`
- **Latest commit:** `1cdea18`

## Architecture

### Model (Wan2.2-TI2V-5B)
- **DiT:** dim=3072, 24 heads, 30 transformer layers, ffn_dim=14336
- **VAE:** Wan2.2_VAE (CausalConv3d-based, z_dim=48, temporal stride=4, spatial stride=16)
- **T5:** UMT5-XXL encoder for text conditioning
- **VAE stride:** (4, 16, 16) — temporal, height, width
- **Patch size:** (1, 2, 2)

### TP-8 Setup
- **8 NeuronCores** (2 NeuronDevices × 4 LNCs each, with `NEURON_LOGICAL_NC_CONFIG=2`)
- **DiT:** TP-sharded across all 8 ranks using `models/tp_utils.py` (from rolling-forcing)
  - 3 heads/rank, ~0.7B params/rank (~1.3GB bf16)
  - Shards: Q/K/V column-parallel, O/down row-parallel, FFN gate/up column-parallel
- **T5:** Loaded on rank 4 only (ND1:NC0), Neuron eager (not compiled)
- **VAE:** Loaded on rank 0 only (ND0:NC0), **currently on CPU eager**
- **torchrun --nproc_per_node=8** for distributed launch

### torch.compile Strategy (DiT only)
Compiled sub-modules on Neuron:
- `patch_embedding`, `text_embedding`, `head`
- `block.ffn` for all 30 blocks

**NOT compiled** (kept in eager):
- `time_embedding`, `time_projection` — receive fp32 sinusoidal input but have bf16 weights (Neuron compiler can't handle mixed-precision here)
- Attention blocks — use NKI kernels directly (see below)

### NKI Flash-Attention Kernels
Located in `kernels/` directory (ported from rolling-forcing):
- **`kernels/cross_attention.py`** — `wan_cross_attn`: for cross-attention (small seq_k=512)
- **`kernels/self_attention.py`** — `wan_flash_self_attn`: for self-attention (large seq_k, padded to 8192 multiple)
- Loaded via `torch_neuronx.nki_hop.wrap_nki()` in `wan/modules/attention.py`
- Self-attention kernel requires seq_k padded to multiples of 8192
- Both kernels expect: q(bs, d, seq_q), k(bs, d, seq_k), v(bs, seq_k, d) with bs=num_heads

### RoPE
Custom `rope_neuron.py` handles rotary position embeddings for video (3D: temporal + spatial).

## Current Configuration
- **frame_num = 81** (10 sec at ~8fps, 3.4 sec at 24fps)
- **Resolution:** 480 × 832 (`max_area = 480 * 832`)
- **Denoising steps:** 20
- **seq_len:** ~8190 tokens (fits NKI 8192 limit)
- **Latent shape:** [48, 21, 30, 52] (z_dim, T_latent, H_latent, W_latent)
- **Guidance scale:** from config (`sample_guide_scale`)
- **Seed:** 42

## Current Performance (81 frames, 480p, 20 steps)
- **Model loading + TP sharding:** ~10s
- **T5 encoding:** ~40s (with Neuron compilation on first call)
- **VAE encode (CPU):** ~8 min (torch.compile compilation overhead on first call was ~8 min when on Neuron)
- **Denoising:** ~170-285s (14s/step × 20 steps)
  - Each step: 2 DiT forward passes (cond + uncond for CFG)
  - NKI self-attn: q(3, 128, 4352→8192), cross-attn: q(3, 128, 4352), k(3, 128, 512)
- **VAE decode (CPU):** ~5 min 
- **Total:** ~15-20 min end-to-end

## VAE Problem — THE KEY ISSUE FOR NEXT THREAD

### Why VAE is on CPU (not Neuron)
The VAE was moved to CPU because **`torch.compile` on the VAE creates 400+ NEFFs** (one per CausalConv3d op, each with its own compiled kernel). This:
1. **Eats ~20GB HBM** on NC4 (rank 0's NeuronDevice) just for model code + scratchpad
2. **Leaves no room for activation tensors** during decode
3. **OOM error:** `Failed to allocate 1.038GB on ND 0:NC 4`

### Memory Layout When VAE Was on Neuron
From the error logs:
```
ND 0 HBM 2: 20.122GB total
  NC 4: 18.862GB used (Model Code: 202MB, Constants: 20MB, Tensors: 14.5GB, Scratchpad: 3.66GB)
  NC 5: 1.260GB used
```
- 14.5GB in tensors (DiT weights) + 3.66GB scratchpad + 202MB model code (compiled NEFFs) = no room for 1GB VAE decode activation

### VAE Architecture (Wan2.2_VAE — `wan/modules/vae2_2.py`)
- Uses `CausalConv3d` extensively (temporal causal convolution)
- Has encoder and decoder paths
- `encode()`: image [C, 1, H, W] → latent [z_dim, 1, H/16, W/16]
- `decode()`: latent [z_dim, T, H/16, W/16] → video [C, F, H, W]
- Scale tensors for normalization during encode/decode
- dtype fix applied: `z = z.to(self.conv2.weight.dtype)` after scale division (was causing NaN on Neuron due to fp32/bf16 mismatch)

### Potential VAE Optimization Approaches
1. **Don't `torch.compile` the whole VAE** — instead compile only specific layers (e.g., the heavy conv layers) to reduce NEFF count
2. **Run VAE on a separate NeuronDevice** — VAE only needs 1 NC, currently sharing ND0 with DiT weights. Could use ND1 after T5 is freed.
3. **Chunk-based decode** — decode temporal chunks separately to reduce peak memory
4. **Mixed precision** — keep VAE in fp32 on CPU but only for decode (encode is fast since it's single-frame)
5. **Offload DiT weights before VAE decode** — free DiT from NC4 HBM, then run VAE decode on Neuron
6. **Profile which CausalConv3d ops are heaviest** — selectively compile only those

### Current Workaround
VAE runs on CPU in float32 (eager mode):
```python
vae.model = vae.model.to(dtype=torch.float32)  # CPU
x0 = [latent.cpu().float()]
videos = vae.decode(x0)
```
This takes ~5 min for 81 frames at 480p — **the biggest bottleneck**.

## File Structure
```
wan2-ti2v-5b/
├── inference_neuron_ti2v.py    # Main inference script
├── setup.sh                     # Entrypoint (deps, model cache, torchrun)
├── rope_neuron.py               # Custom RoPE for 3D video
├── models/
│   └── tp_utils.py              # TP sharding utilities (from rolling-forcing)
├── kernels/
│   ├── cross_attention.py       # NKI cross-attention kernel
│   └── self_attention.py        # NKI self-attention kernel
├── wan/                          # Pre-patched Wan2.2 source
│   ├── __init__.py
│   ├── configs.py
│   ├── modules/
│   │   ├── __init__.py           # Exports flash_attention (alias)
│   │   ├── attention.py          # NKI-aware attention dispatch
│   │   ├── model.py              # WanModel (DiT)
│   │   ├── t5.py                 # T5 encoder
│   │   ├── vae2_1.py             # Wan2.1 VAE
│   │   ├── vae2_2.py             # Wan2.2 VAE (with dtype fix)
│   │   └── tokenizers.py
│   ├── distributed/
│   └── utils/
│       ├── utils.py              # best_output_size, masks_like
│       └── fm_solvers_unipc.py   # FlowUniPC scheduler
└── wan_requirements.txt          # Python deps (flash_attn/torchaudio stripped)
```

## K8s Job Configuration
```yaml
image: 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-d1c940d:latest
nodeSelector: trn2.48xlarge
resourceClaims: l-trn2 (DRA)
resources: 44 CPU, 440Gi memory
env:
  NEURON_LOGICAL_NC_CONFIG: "2"
  NEURON_CC_FLAGS: "--model-type=transformer"
  TP_DEGREE: "8"
  T5_RANK: "4"
  TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS: "0"
volumes: S3-backed PVC at /var/mdl
```

## Key Decisions & Lessons Learned
1. **`flash_attention` alias needed** — `wan/modules/__init__.py` imports `flash_attention` but our rewritten `attention.py` named it `attention`. Added `flash_attention = attention` alias.
2. **shutil.copy fails on S3 FUSE** — `PermissionError` on `copymode()`. Fixed by using bash `cp` in `setup.sh` instead.
3. **torch.compile VAE = 400+ NEFFs = OOM** — each CausalConv3d becomes separate compiled kernel. Moved VAE to CPU.
4. **T5 torch.compile removed** — single-use model, compilation overhead > runtime. Runs eager on Neuron.
5. **time_embedding/time_projection in eager** — fp32 sinusoidal input + bf16 weights = Neuron compiler crash.
6. **imageio instead of torchvision.io.write_video** — avoids PyAV dependency.
7. **VAE dtype fix** — `z.to(self.conv2.weight.dtype)` after scale division in vae2_2.py to prevent NaN.

## Output
- Video saved to `/tmp/wan2_ti2v_output.mp4`
- Copied to `/var/mdl/wan2_2_ti2v/outputs/` via bash cp
- Format: H.264 (libx264) via imageio-ffmpeg, 24fps
