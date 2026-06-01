# Wan2.2-TI2V-5B on AWS Trainium

Text+Image-to-Video generation (81 frames, 480p) using the Wan2.2-TI2V-5B diffusion model, optimized for AWS Trainium (trn2) with tensor parallelism and sequence parallelism.

## Benchmark Results

**Single NeuronDevice** (`s-lnc1-trn2`, SP=2 TP=4, 8 NeuronCores):

| | Denoise | s/step | VAE decode | Total |
|---|---|---|---|---|
| Run 1 (cold/compile) | 116.0s | 11.6s | 294.1s | 410.2s |
| **Run 2 (warm)** | **48.5s** | **4.8s** | **31.7s** | **80.1s** |

- 81 frames, 704x544, 10 denoising steps
- DiT: NST flash self-attention NKI kernel + NKI cross-attention kernel
- VAE: PyTorch eager on Neuron (NKI kernels pending SDK migration)

## Architecture

Three models run in a pipeline:

| Model | Role | Placement |
|-------|------|-----------|
| **T5 (UMT5-XXL)** | Text encoder | Single rank (rank 2) |
| **DiT (WanModel)** | 30-layer diffusion transformer | TP-sharded across all ranks |
| **VAE (Wan2.2_VAE)** | CausalConv3d encoder/decoder | Rank 0, PyTorch eager |

**DiT specs:** dim=3072, 24 heads, 30 layers, ffn_dim=14336, ~5.3B params total, ~1.32B per rank at TP=4.

### Parallelism Strategy (SP=2 TP=4)

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

- Kubernetes cluster with trn2 nodes (tested on trn2.48xlarge)
- DRA resource claims: `s-lnc1-trn2` (single ND) or `m-lnc1-trn2` (2 NDs)
- S3-backed PVC mounted at `/var/mdl` for model weight caching
- Secrets: `hf-token` (HuggingFace), `github-token` (repo clone)

### Run Benchmark (Single NeuronDevice)

```bash
kubectl apply -f wan2-ti2v-5b-job.yaml
kubectl logs -f job/wan2-ti2v-5b
```

Configuration: SP=2, TP=4, 10 denoising steps, 81 frames, NST self-attention kernel.

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
| `TP_DEGREE` | 8 | DiT tensor parallelism degree |
| `SP_DEGREE` | 1 | Sequence parallelism degree (2 for SP+TP on single ND) |
| `T5_RANK` | 2 | Which rank hosts T5 encoder |
| `VAE_TP_DEGREE` | 1 | VAE channel parallelism |
| `USE_NKI_KERNELS` | 1 | Enable NKI kernels for DiT attention |
| `USE_NKI_VAE` | 0 | Enable NKI kernels for VAE (disabled pending SDK migration) |
| `USE_NST_SELF_ATTN` | 1 | Use Neuron Science Team self-attention kernel |
| `NUM_STEPS` | 10 | Denoising steps |
| `NUM_RUNS` | 2 | Benchmark iterations (run 1=cold, run 2=warm) |
| `FRAME_NUM` | 81 | Output video frames |
| `MODEL_PATH` | /tmp/Wan2.2-TI2V-5B | Path to model weights |

## Project Structure

```
wan2-ti2v-5b/
├── inference_neuron_ti2v.py      # Benchmark script (torchrun entry point)
├── serve_ti2v.py                 # FastAPI server with TP-8
├── setup.sh                      # Container entrypoint (deps + model + launch)
├── wan2-ti2v-5b-job.yaml         # K8s job: SP=2 TP=4, s-lnc1-trn2
├── wan2-ti2v-5b-deploy.yaml      # K8s deployment: TP=8, serving
├── models/
│   ├── tp_utils.py               # TP sharding (Column/RowParallelLinear)
│   ├── parallel_state.py         # SP/TP process group registry
│   ├── dit_attention_sp.py       # SP-aware bidirectional self-attention
│   └── vae_tp.py                 # VAE channel TP (optional)
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
- **torch.compile(backend='neuron')** for DiT FFN/embeddings/head; time_embedding stays eager (fp32 input, bf16 weights)
- **All collectives on main thread** -- the serving FastAPI thread queues requests to main thread which owns Neuron collective signatures
- **All-reduce chunking** -- payloads chunked to <=8MB to avoid NRT size-limit rejections at TP=8
- **NKI SDK (new)** -- uses `import nki` (bare), `@nki.jit` decorators, `@wrap_nki` for PyTorch integration, dst-style `nisa.nc_matmul(dst, stationary, moving)`

## Model Weights

First run downloads from HuggingFace (`Wan-AI/Wan2.2-TI2V-5B`) and caches as tar on the S3-backed PVC. Subsequent runs extract from cache (~30s).

Requires `HF_TOKEN` secret for gated model access.
