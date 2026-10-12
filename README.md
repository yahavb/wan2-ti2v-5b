# Wan2.2-TI2V-5B on AWS Trainium

Text+Image-to-Video generation using the Wan2.2-TI2V-5B diffusion model on AWS Trainium (trn2), with a block-compiled tensor-parallel DiT (`models/fast_dit.py`).

## Benchmark Results

All numbers: one trn2 NeuronDevice (8 physical NeuronCores), 10 denoising steps, CFG on,
warm run (after compile). Same prompt/image/seed. Fast DiT rows measured 2026-10-10 on the `fast-dit` branch (now merged to `main`);
original-path rows are the earlier measurements.

### 704x544, 81 frames (7854 DiT tokens)

| Path | Layout | Claim | Denoise | s/step | VAE decode | Total | First run (compile) |
|---|---|---|---|---|---|---|---|
| **Fast DiT** (`DIT_FAST=1`) | **TP=8, LNC1** | `s-lnc1` | **19.1s** | **1.91** | 17.7s | **36.8s** | 261.9s |
| Fast DiT (`DIT_FAST=1`) | TP=4, LNC2 | `s-lnc2` | 25.5s | 2.55 | 10.8s | 36.4s | 286.8s |
| Original eager path (`DIT_FAST=0`) | TP=8, LNC1 | `s-lnc1-trn2` | 44.3s | 4.4 | 32.3s | 76.6s | — |
| Original eager path, SP | SP=2 TP=4, LNC1 | `s-lnc1-trn2` | 50.8s | 5.1 | 31.6s | 82.4s | — |

- **Fast DiT is 2.3x faster on denoise than the original path on the same layout** (4.4 -> 1.91 s/step).
- **TP=8 LNC1 beats TP=4 LNC2 on the DiT by 25%** (1.91 vs 2.55 s/step) on the same physical device.
- **VAE favors LNC2** (10.8s vs 17.7s): it runs on one logical core, which is 2 physical cores at LNC2
  and 1 at LNC1. End to end the two layouts tie at 10 steps; TP=8 LNC1 wins as step count grows.
- Output video validated visually for the TP=4 LNC2 fast path.

### 368x480, 21 frames (generated at 384x480, center-cropped; 1080 DiT tokens)

| Path | Layout | Denoise | s/step | VAE decode | Total | First run (compile) |
|---|---|---|---|---|---|---|
| Fast DiT (`DIT_FAST=1`) | TP=8, LNC1 | 1.5s | 0.15 | 2.4s | **3.9s** | 83.5s |

With the cross-attention K/V cache (`1b18e22`) denoise drops to 1.4s (0.14 s/step).
With the 8-rank tiled VAE decode and the RMS_norm rewrite (`12fd9ce`, `VAE_TILED=1`), VAE decode drops to 1.7s and the total to **3.2s** (seam quality: check the video).

Set with `HEIGHT=368 WIDTH=480 FRAME_NUM=21` (sizes that are not multiples of 32 are generated
at the next multiple of 32 and center-cropped after decode).

### What the fast DiT path does (`models/fast_dit.py`)

- Each of the 30 transformer blocks is **one compiled graph** with the TP all-reduces inside it
  (functional collectives). The original path compiled only the FFN and ran attention, norms,
  modulation and collectives eagerly.
- **CFG batched**: cond and uncond run as batch=2 in one forward, one model call per step.
- **RoPE cos/sin built once** on device (the original rebuilt them on CPU twice per block per forward).
- **Per-frame time modulation**: TI2V has only two distinct timesteps (image frame = 0, rest = t),
  so modulation is computed for latent frames instead of every token. Exact, not an approximation.
- **Text embedding and cross-attention K/V once** per generation (the text context is fixed across steps).
- **Attention via SDPA** (TorchNeuron's built-in flash kernel; LNC2-aware). `FAST_ATTN_IMPL=nki`
  uses this repo's NKI kernels instead (single-program, so one physical core per logical core at LNC2).

## Architecture

Three models run in a pipeline:

| Model | Role | Placement |
|-------|------|-----------|
| **T5 (UMT5-XXL)** | Text encoder | Single rank (rank 2) |
| **DiT (WanModel)** | 30-layer diffusion transformer | TP-sharded across all ranks; fast path compiles each block as one graph |
| **VAE (Wan2.2_VAE)** | CausalConv3d encoder/decoder | Rank 0, PyTorch eager |

**DiT specs:** dim=3072, 24 heads, 30 layers, ffn_dim=14336, ~5.3B params total; ~1.32B per rank at TP=4 (6 heads/rank), ~0.66B at TP=8 (3 heads/rank).

### Legacy: SP=2 TP=4 (original path only, `DIT_FAST=0`)

8 ranks on a single NeuronDevice (LNC1 = 2 NCs/ND x 4 NDs):

- **TP groups:** [0,1,2,3] and [4,5,6,7] -- heads split 24/4 = 6 per rank
- **SP groups:** [0,4], [1,5], [2,6], [3,7] -- sequence split in half
- Each rank computes attention for L/2 query tokens against full L key tokens
- K/V AllGathered across SP group, Q stays local -- halves attention compute per rank

### NKI Kernels

Custom Neuron Kernel Interface kernels for DiT attention:

| Kernel | File | Description |
|--------|------|-------------|
| NST self-attention | `kernels/self_attention_nst.py` | Flash attention with online softmax, `actual_seqlen_k` masking, no identity/mask tensors needed |
| Cross-attention | `kernels/cross_attention.py` | Single-pass flash attention for T5 context (512 tokens) |
| RoPE | `kernels/rope.py` | 3D rotary position embeddings |
| KV cache copy | `kernels/kv_cache_copy.py` | DMA-based tensor copy via `nki_op` |

Supporting infrastructure: `kernels/nkilib_compat.py`, `kernels/nkilib_modular_allocator.py`, `kernels/nkilib_tensor_view.py`

## Deployment

### Prerequisites

- Kubernetes cluster with trn2 nodes
- DRA resource claims: `s-lnc1` (1 device, 8 logical cores, TP=8) or `s-lnc2` (1 device, 4 logical cores, TP=4)
- Secret `github-token` (repo clone). `hf-token` is optional (the model is public).
- Optional S3-backed PVC at `/var/mdl` for weight caching; without it weights download from Hugging Face each run.

### Run Benchmark

```bash
kubectl apply -f wan2-ti2v-5b-tp8-job.yaml   # TP=8 LNC1 (fastest DiT)
kubectl apply -f wan2-ti2v-5b-job.yaml       # TP=4 LNC2
kubectl logs -f job/wan2-ti2v-5b-tp8
```

Both jobs clone `main` and run `scripts/prof/bench_profile.sh`:
clean pass (`CLEAN_RUNS` runs; run 1 compiles, the rest are the latency), then a short profiled pass
with the Neuron runtime inspector on rank 0, then per-graph analysis. Results go to
`/tmp/results/<run>.tar.gz` in the pod, which stays up for `KEEP_ALIVE_SECONDS` so you can
`kubectl cp` the tarball and `output_clean.mp4` (the commands are printed at the end of the log).

### Run Serving (Multi-NeuronDevice)

```bash
kubectl apply -f wan2-ti2v-5b-deploy.yaml
```

FastAPI server on port 8000:
- `POST /generate` -- `{"prompt": "...", "image_url": "...", "num_steps": 10, "seed": 42}` returns `{"video": "<base64 mp4>", "execution_time": ..., "frames": 81}`
- `GET /health` -- liveness
- `GET /readiness` -- model warmup complete

### DLC Image

```
421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TP_DEGREE` | 8 | DiT tensor parallelism degree (= torchrun ranks) |
| `SP_DEGREE` | 1 | Sequence parallelism (original path only; must be 1 with `DIT_FAST=1`) |
| `DIT_FAST` | 1 | Block-compiled DiT with in-graph all-reduces + batched CFG; 0 = original eager path |
| `FAST_ATTN_IMPL` | sdpa | `sdpa` (TorchNeuron flash kernel), `nki` (this repo's kernels), or `constmax` (odyssey-trainium const-max self-attention, `kernels/attention_const_max.py`) |
| `CONST_MAX_SOFTMAX_MAX` | 15.0 | Constant softmax max for `constmax` (scaled-score units) |
| `CONST_MAX_LAYERS`, `CONST_MAX_HARD_LIMIT` | auto, 70 | `auto`: const-max only on layers whose worst-case score (from QK-norm weights) stays under the kernel's clip; others use sdpa. Or `all`, or a comma list |
| `FAST_FULLGRAPH` | 0 | 1 = fail on any graph break inside a block |
| `FAST_ALLREDUCE_CHUNK_BYTES` | 0 | Chunk in-graph all-reduces (0 = off) |
| `HEIGHT`, `WIDTH` | unset | Output size; unset = derived from the input image aspect within 480x832 area |
| `FRAME_NUM` | 81 | Output frames (must be 4k+1) |
| `NUM_STEPS` | 10 | Denoising steps |
| `NUM_RUNS` | 2 | Iterations per process (run 1 compiles) |
| `T5_RANK` | 2 | Rank hosting the T5 encoder (clamped to world size) |
| `VAE_TP_DEGREE` | 1 | VAE channel parallelism |
| `USE_NKI_KERNELS` | 1 | Load NKI kernels for DiT attention (used by `FAST_ATTN_IMPL=nki` and the original path) |
| `USE_NKI_VAE` | 0 | NKI kernels for VAE (disabled pending SDK migration) |
| `VAE_FULLGRAPH` | 0 | fullgraph=True for the VAE compile |
| `VAE_TILED`, `VAE_TILE_OVERLAP` | 0, 8 | Decode overlapping spatial tiles on all ranks, blend on rank 0 (`models/vae_tiled.py`); overlap in latent units |
| `VAE_COMPILE` | 0 | Compile the VAE decode via `models/vae_static.py` (fixed-shape causal cache, 2 graphs; ported from vllm-omni-neuron). An earlier per-chunk compile of the original decoder **measured slower**: 11.5s warm vs 2.4s eager at 368x480x21, 58 min compile. Keep 0 |
| `USE_NST_SELF_ATTN` | 1 | NST self-attention kernel (original path) |
| `CLEAN_RUNS`, `PROFILE_STEPS`, `PROFILE_RANKS`, `MAX_PAIRS` | 3, 2, 0, 3 | Benchmark/profiling harness (`scripts/prof/bench_profile.sh`) |
| `KEEP_ALIVE_SECONDS` | 86400 (jobs) | Keep the pod up after the run for `kubectl cp` |
| `MODEL_PATH` | /tmp/Wan2.2-TI2V-5B | Path to model weights |

## Project Structure

```
wan2-ti2v-5b/
├── inference_neuron_ti2v.py      # Benchmark script (torchrun entry point)
├── serve_ti2v.py                 # FastAPI server with TP-8
├── setup.sh                      # Container entrypoint (deps + model + launch)
├── wan2-ti2v-5b-job.yaml         # K8s job: TP=4, s-lnc2 (LNC2)
├── wan2-ti2v-5b-tp8-job.yaml     # K8s job: TP=8, s-lnc1 (LNC1)
├── wan2-ti2v-5b-deploy.yaml      # K8s deployment: TP=8, serving
├── models/
│   ├── fast_dit.py               # Fast DiT: block-compiled, batched CFG
│   ├── tp_utils.py               # TP sharding (Column/RowParallelLinear)
│   ├── parallel_state.py         # SP/TP process group registry
│   ├── dit_attention_sp.py       # SP-aware bidirectional self-attention
│   └── vae_tp.py                 # VAE channel TP (optional)
├── scripts/prof/                 # 3-run benchmark + profiling harness
├── kernels/
│   ├── self_attention_nst.py     # NST flash self-attention (primary)
│   ├── cross_attention.py        # NKI cross-attention
│   ├── rope.py                   # NKI RoPE
│   ├── kv_cache_copy.py          # NKI DMA cache copy
│   ├── self_attention.py         # Legacy mask-based self-attention
│   ├── vae_conv2d.py             # VAE conv2d kernels (pending SDK fix)
│   ├── vae_attention.py          # VAE attention kernel (pending SDK fix)
│   └── nkilib_*.py               # NST kernel infrastructure
└── wan/                          # Pre-patched Wan2.2 source
    ├── modules/
    │   ├── model.py              # WanModel (DiT)
    │   ├── attention.py          # NKI-aware attention dispatch
    │   ├── vae2_2.py             # Wan2.2 VAE with NKI dispatch
    │   └── rope_neuron.py        # Neuron-compatible RoPE
    ├── configs/
    │   └── wan_ti2v_5B.py        # TI2V-5B config
    └── utils/
        └── fm_solvers_unipc.py   # Flow-matching scheduler
```

## Neuron-Specific Constraints

- **No float64/complex types** on device -- custom `rope_neuron.py` avoids `torch.polar`/`view_as_complex`
- **torch.compile(backend='neuron')**: fast path compiles each DiT block as one graph with in-graph all-reduces; the original path compiles only FFN/embeddings/head
- **All collectives on main thread** -- the serving FastAPI thread queues requests to main thread which owns Neuron collective signatures
- **All-reduce chunking** (original eager path) -- payloads chunked to <=8MB to avoid NRT size-limit rejections at TP=8; not needed in-graph
- **NKI SDK (new)** -- uses `import nki` (bare), `@nki.jit` decorators, `@wrap_nki` for PyTorch integration, dst-style `nisa.nc_matmul(dst, stationary, moving)`

## Model Weights

First run downloads from Hugging Face (`Wan-AI/Wan2.2-TI2V-5B`, public) and, if the S3-backed PVC is mounted, caches it as a tar there. Subsequent runs extract from cache (~30s).
