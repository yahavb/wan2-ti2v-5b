# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wan2.2-TI2V-5B (Text+Image-to-Video, 5 billion params) inference optimized for AWS Trainium/Inferentia (Neuron). The pipeline generates 81-frame 480p videos from a text prompt and reference image using a DiT (Diffusion Transformer) with tensor parallelism across 8 NeuronCores.

## Architecture

Three models run in a pipeline:
- **T5 (UMT5-XXL)**: Text encoder on a single rank (T5_RANK, default rank 2 for serving)
- **DiT (WanModel)**: 30-layer transformer (dim=3072, 24 heads, ffn_dim=14336), TP-8 sharded across all ranks
- **VAE (Wan2.2_VAE)**: CausalConv3d encoder/decoder on rank 0, with NKI kernel fusion for Neuron acceleration

TP sharding pattern (in `models/tp_utils.py`):
- Q/K/V → ColumnParallelLinear (split by heads: 3 heads/rank at TP=8, 6 at TP=4)
- O/FFN-down → RowParallelLinear (all-reduce after matmul)
- FFN-up → ColumnParallelLinear
- QK norms → TPRMSNorm (all-reduce sum-of-squares for global RMS)

SP+TP layout (in `models/parallel_state.py`, `models/dit_attention_sp.py`):
- SP=2, TP=4 on single ND: 8 ranks, TP groups [0-3]/[4-7], SP groups [0,4]/[1,5]/[2,6]/[3,7]
- Each rank computes Q for L/SP tokens, AllGathers full K/V across SP for attention
- Reduces per-rank attention compute by SP factor while keeping communication low

## Key Entry Points

- `inference_neuron_ti2v.py` — Benchmark script: runs NUM_RUNS iterations, prints timing table
- `serve_ti2v.py` — FastAPI server (POST /generate, GET /health, GET /readiness). Rank 0 runs HTTP server thread; all Neuron collectives happen on main thread via request queue
- `setup.sh` — Container entrypoint: installs deps, downloads/caches model weights, launches torchrun

## Running

```bash
# Benchmark TP=8 (on Neuron instance, 2 NDs with LNC2)
export MODEL_PATH=/tmp/Wan2.2-TI2V-5B
torchrun --nproc_per_node=8 --master_port=29500 inference_neuron_ti2v.py

# Benchmark SP=2 TP=4 (on single ND with LNC1, s-lnc1-trn2)
SP_DEGREE=2 TP_DEGREE=4 USE_NST_SELF_ATTN=1 \
  torchrun --nproc_per_node=8 --master_port=29500 inference_neuron_ti2v.py

# Serving
torchrun --nproc_per_node=8 --master_port=29500 serve_ti2v.py

# VAE kernel tests (requires Neuron runtime)
python test_vae_kernels.py
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TP_DEGREE` | 8 | DiT tensor parallelism degree |
| `SP_DEGREE` | 1 | Sequence parallelism degree (2 for SP+TP on single ND) |
| `T5_RANK` | 8 (bench) / 2 (serve) | Which rank hosts T5 |
| `VAE_TP_DEGREE` | 1 | VAE channel parallelism (1=NKI, 2=channel-split) |
| `USE_NKI_KERNELS` | 0 | Enable NKI kernels for DiT attention |
| `USE_NKI_VAE` | 1 | Enable NKI kernels for VAE conv/attention |
| `USE_NST_SELF_ATTN` | 0 | Use Neuron Science Team self-attention kernel (no mask/identity needed) |
| `NUM_STEPS` | 10 | Denoising steps |
| `NUM_RUNS` | 2 | Benchmark iterations (run 1=cold/compile, run 2=warm) |
| `FRAME_NUM` | 81 | Output video frames |
| `NEURON_LOGICAL_NC_CONFIG` | 2 | Logical NeuronCores per device (set in k8s yaml) |

## NKI Kernels (`kernels/`)

Custom Neuron Kernel Interface (NKI) kernels replace torch.compile for the VAE (which would generate 400+ NEFFs and OOM):
- `vae_conv2d.py`: Spatial conv2d via shifted-matmul approach (no im2col). K=1 (pointwise) and K=3 (9 shifted matmuls)
- `vae_attention.py`: Single-head self-attention for VAE attention blocks
- `self_attention_nst.py`: Neuron Science Team flash self-attention (preferred for SP mode — handles actual_seqlen_k, no mask/identity)
- `self_attention.py`: Original mask-based DiT self-attention (fallback)
- `cross_attention.py`: DiT cross-attention kernel
- `rope.py`, `kv_cache_copy.py`: Supporting DiT kernels
- `nkilib_compat.py`, `nkilib_modular_allocator.py`, `nkilib_tensor_view.py`: NST kernel infrastructure

The VAE uses NKI when `USE_NKI_VAE=1` (loaded in `wan/modules/vae2_2.py` via `torch_neuronx.nki_hop.wrap_nki`).

## Neuron-Specific Constraints

- No float64, complex types, `torch.polar`, or `view_as_complex` on device → custom `rope_neuron.py`
- `torch.compile(backend='neuron')` used for DiT submodules but NOT time_embedding/time_projection (fp32 input with bf16 weights)
- All `dist.broadcast`/`dist.all_reduce` must happen from the main thread (same thread as warmup) — the server uses a queue to route requests from FastAPI thread to main thread
- All-reduce payloads chunked to ≤8MB (`MAX_ALLREDUCE_BYTES`) to avoid NRT size-limit rejections at TP=8
- `@torch.compiler.disable` on `all_reduce_sum` forces graph breaks between compiled FFN NEFFs and collective ops

## Deployment

Kubernetes on trn2 instances. Model weights cached as tar on S3-backed PVC at `/var/mdl/wan2_2_ti2v/`.
- `wan2-ti2v-5b-deploy.yaml` — Serving (TP=8, m-lnc1-trn2)
- `wan2-ti2v-5b-job.yaml` — Benchmark (TP=8, m-lnc1-trn2, 2 NDs with LNC2)
- `wan2-ti2v-5b-sp2-job.yaml` — Benchmark (SP=2 TP=4, s-lnc1-trn2, single ND) with Neuron profiling
- `wan2-ti2v-5b-explorer-job.yaml` — Post-hoc Neuron Explorer analysis of profiled runs

## The `wan/` Directory

Pre-patched fork of Alibaba's Wan2.2 source. Key modifications:
- `wan/modules/model.py`: Uses `rope_neuron` instead of complex-valued RoPE
- `wan/modules/vae2_2.py`: NKI kernel dispatch for conv2d and attention
- `wan/modules/attention.py`: NKI-aware attention dispatch
- `wan/utils/fm_solvers_unipc.py`: Flow-matching scheduler for denoising
