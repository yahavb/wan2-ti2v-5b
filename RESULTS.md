# Benchmark log

All runs: one trn2 NeuronDevice, 10 denoising steps, CFG on, same prompt/image/seed.
Numbers are warm runs (after compile). Measured 2026-10-10 to 2026-10-12.

## `loop_engineering` branch (2026-10-28), 704x544, 81 frames, TP over 8 ranks LNC1

Image `concourse-release-0461d3b:latest` pushed 2026-10-09 (torch 2.14.0, torch-neuronx 2.14.3,
neuronx-cc 2.28.4568, nki 0.7.0), run in Docker on a trn2.48xlarge, one device. Warm run 3 of 3.
"Accuracy" = rel-RMS of the step-0 CFG noise prediction against an fp32 CPU run of the same model on
the same input (`scripts/validate/ref_fp32.py`, on a `DUMP_DIR` dump): every bf16 configuration lands at ~0.048, so this checks
that a change adds no error beyond bf16. Videos checked visually on frames 20/50/80.

| Change (cumulative) | s/step | Denoise | VAE | Total | Accuracy |
|---|---|---|---|---|---|
| `main@feb083d` (const-max 22/30 layers + sdpa, TP=8, batch-2 CFG) | 1.20 | 12.0s | 4.3s | 16.2s | 0.0474 |
| exact SDPA on all 30 layers (reference) | 1.92 | 19.2s | — | — | 0.0485 |
| nkilib `attention_cte` self-attention, all 30 layers | 0.94 | 9.4s | 4.4s | 13.7s | — |
| + `attention_cte` cross-attention | 0.90 | 9.0s | 4.1s | 13.2s | 0.0482 |
| + CFG-parallel: 2 groups x TP4, batch 1 each | 0.76 | 7.6s | 4.3s | 11.9s | 0.0483 |
| + Megatron sequence parallelism (AG/RS around the TP matmuls) | 0.68 | 6.8s | 4.4s | 11.1s | 0.0477 |
| + cross-attention local to the token shard (all heads, no collectives) | 0.64 | 6.4s | 4.4s | 10.8s | 0.0484 |
| + VAE: RMS_norm + SiLU as one compiled graph (`VAE_FUSE_NORM=1`) | 0.64 | 6.4s | 3.5s | 10.0s | 0.0487* |
| + CFG groups on alternating cores (`CFG_GROUPS=interleave`) | **0.62** | **6.2s** | **3.5s** | **9.6s** | 0.0487* |

Env for the last row: `CFG_PARALLEL=1 CFG_GROUPS=interleave FAST_SP=1 FAST_LOCAL_CROSS=1
FAST_ATTN_IMPL=cte FAST_CROSS_IMPL=cte VAE_FUSE_NORM=1`.
\* The fused norm also runs in the VAE encoder, so the conditioning latent (and the step-0 input)
changes slightly; these rows are scored against an fp32 reference computed on their own input.
`CFG_GROUPS=interleave` gives bit-identical output to the contiguous layout.

VAE accuracy, same final latent, against an exact fp32 CPU decode without tiling
(`scripts/validate/vae_cpu_ref.py`): tiled bf16 decode (overlap 4) 44.67 dB, with
`VAE_FUSE_NORM=1` 44.69 dB; fused vs unfused 55.75 dB. Overlap 3: 43.74 dB (3.01 s), overlap 2:
42.24 dB (2.54 s) vs 3.10 s for overlap 4 -- not taken, the speed costs 1-2.5 dB.

Per-call attention microbench (one core, 6 heads x 7854 tokens, d=128, layer-25 QK-norm gammas):
sdpa 18.2 ms, const-max 5.8 ms, `attention_cte` 5.3 ms, all at bf16-level error (rel-RMS 0.0055
vs fp32). Per-block (TP4, batch 1): 29.4 ms (TP8 batch 2) -> 24.5 (CFG-parallel) -> 21.9 (+SP) -> 20.6 ms (+local cross).

Where the SP block's 20.6 ms goes (contiguous groups): collectives 6.2 ms (block without them:
14.4 ms), QK-norm 0.6 ms, RoPE 0.5 ms. Interleaving the CFG groups gives 19.8 ms. Per denoising
step the host side (D2H, CPU UniPC step, H2D) is 4-6 ms of ~630 ms (`STEP_TIMING=1`).

Tried and rejected (block bench unless noted): chunking the in-graph all-reduces (29.4 -> 36.4 ms),
`NEURON_COMPILER_OPT_LEVEL=-O3` (no change), frame-chunked sequence split inside the block
(24.4 -> 24.8/27.1 ms), pre-transposed weights (25.7 ms), fused QKV projection and a [real|imag]
RoPE layout (both within 0.1 ms), TP8 + SP instead of CFG-parallel (23.7 vs 21.9 ms), runtime
collective-algorithm knobs `NEURON_RT_DBG_{MESH,RDH,KANGARING}_CC` and friends (20.59-20.72 vs
20.62 ms), nkilib `allgather_compute_matmul` (fails at our 1964-row shard; 3.95 vs 3.73 ms at 2048
rows). The compiler does not overlap an in-graph all-gather with independent compute
(`scripts/validate/overlap_bench.py`: 2.95 + 3.65 -> 6.53 ms together). VAE decoding 2 or 4 latent
frames per decoder call is 11-18% faster per tile on an empty core (`VAE_TCHUNK`, removed) but
runs out of HBM next to the resident DiT (drain-and-retry stalls, then OOM) and compiles 7-25 min.

## 704x544, 81 frames (7854 DiT tokens)

| Commit | Change | Layout | s/step | Denoise | VAE | Total | Video |
|---|---|---|---|---|---|---|---|
| (main, before) | original eager DiT | TP=8 LNC1 | 4.4 | 44.3s | 32.3s | 76.6s | ok |
| `ce390d3` | fast DiT: block-compiled, batched CFG | TP=4 LNC2 | 2.55 | 25.5s | 10.8s | 36.4s | ok |
| `ce390d3` | fast DiT | TP=8 LNC1 | 1.91 | 19.1s | 17.7s | 36.8s | not checked |
| `2e4a6a6` | + cross-attn K/V cache + tiled VAE | TP=8 LNC1 | 1.92 | 19.2s | 9.0s | 28.2s | not checked |
| `b8d5cbd` | + const-max self-attention on 22/30 layers (1 head per kernel call), rest sdpa | TP=8 LNC1 | 1.19 | 11.9s | 9.0s | 21.0s | perfect |
| `de16d5d` | + VAE tiles weighted on device, collected via /dev/shm; tile overlap 4 | TP=8 LNC1 | **1.19** | **11.9s** | **6.0s** | **17.9s** | amazing |
| `679701a` | + nkilib conv3d for VAE causal 3x3x3 convs (C>=96, D_out>=2), T5 cache freed | TP=8 LNC1 | **1.19** | **11.9s** | **5.4s** | **17.3s** | conv rel_err <= 1e-4 |
| `99f92da` | + upsample convs via nkilib (D=1), 2x upsample as repeat_interleave | TP=8 LNC1 | 1.19 | 11.9s | 5.2s | 17.2s | 2D rel_err 0.0000 |
| `08921e3` | + rank-0 CPU threads 16 (torchrun default 1), in-place uint8 convert | TP=8 LNC1 | **1.19** | **11.9s** | **4.2s** | **16.1s** | — **beats A100 16.92s** |

VAE breakdown at `08921e3`: per-core decode 3.8s, collect+sum 0.14s, convert 0.02s, mp4 write 0.21s.

VAE breakdown at `de16d5d`: per-core decode 4.5s, collect+sum 0.4s, convert+mp4 write 1.1s (was 6.1s / 2.3s gather+blend / 0.9s at `21759b3`, overlap 8).

Reference (5B, same shape class, from the 2026-10-07 comparison): H100 compiled 0.498 s/step, 8.02s total;
A100 compiled 1.143 s/step, 16.92s total; earlier Trainium2 TP8 3.971 s/step, 58.68s total.

## Comparison with GPUs (Wan2.2-TI2V-5B, 704x544, 81 frames, 10 steps)

| Platform | s/step | Denoise | VAE + output | Total | Trainium2 (current main) vs this |
|---|---|---|---|---|---|
| H100 compiled, BF16 VAE + autotuning | 0.498 | 4.98s | 3.01s | 8.02s | 2.2x slower total |
| A100 compiled, BF16 VAE + autotuning | 1.143 | 11.43s | 5.48s | 16.92s | **1.06x slower total; denoise and VAE near parity** |
| Trainium2 earlier (TP8/SP1) | 3.971 | 39.71s | 18.86s | 58.68s | 3.3x faster |
| **Trainium2 current main (TP8 LNC1, one device)** | **1.19** | **11.9s** | **6.0s** | **17.9s** | — |

GPU numbers are from the 2026-10-07 comparison table (sequential CFG); Trainium2 uses batched CFG.
Job: `wan2-ti2v-5b-tp8-job.yaml` (FRAME_NUM=81, HEIGHT/WIDTH unset).

## 368x480, 21 frames (generated 384x480, cropped; 1080 DiT tokens), TP=8 LNC1

| Commit | Change | s/step | Denoise | VAE | Total | Video |
|---|---|---|---|---|---|---|
| `50db102` | fast DiT at this shape | 0.15 | 1.5s | 2.4s | 3.9s | not checked |
| `1b18e22` | + cross-attention K/V cached per prompt | 0.14 | 1.4s | 2.4s (eager) | 3.8s (est.) | — |
| `1b18e22` | VAE torch.compile of original decoder (`VAE_COMPILE=1`) | 0.14 | 1.4s | 11.5-12.9s | 12.9s | — (58 min compile; dropped) |
| `ccd1e2c` | VAE static-cache compile, 2 graphs (vllm-omni port) | 0.14 | 1.4s | 12.8-14.2s | 14.2s | — (20.6 min compile; dropped) |
| `12fd9ce` | VAE eager, tiled over 8 ranks + RMS_norm rewrite | 0.14 | 1.4s | **1.7s** | **3.2s** | perfect |
| `71f188d` | const-max self-attention, all 30 layers | 0.13 | 1.3s | 1.7s | 3.0s | **bad after frame 0** |
| `dae9160` | const-max hybrid, 22/30 layers (rest sdpa) | 0.14 | 1.4s | 1.7s | 3.1s | perfect |

**Current default on `main`:** fast DiT + const-max hybrid self-attention + cross-attn K/V cache + tiled VAE, TP=8 LNC1:
**1.19 s/step, 17.9s total** at 704x544x81; 0.14 s/step, 3.1s total at 368x480x21.

## Notes

- TP=8 LNC1 beat TP=4 LNC2 on the DiT by 25% (1.91 vs 2.55 s/step); the VAE is faster at LNC2
  when it runs on one core.
- Compiling the VAE (two approaches) was ~5x slower than eager after compile. Likely cause:
  compiler-generated convs; vllm-omni uses NKI conv kernels instead (not ported).
- TorchNeuron reads `NEURON_CC_FLAGS` once per process; per-graph compiler flags are not possible.
- const-max kernel: on random inputs it matches exact attention (rel-rms ~0.005, partial
  128-tiles fine). On real activations, all-layer const-max degraded frames; restricting it to
  layers whose worst-case score from the QK-norm weights is <= 70 (the kernel's clip) fixed
  quality but removed the speedup at 1080 tokens. At 7854 tokens it is the big win: 1.92 -> 1.19 s/step (-38%),
  after calling the kernel one head at a time (all heads at once overflowed SBUF and crashed neuronx-cc).
- Layer choice must be identical on every rank (max over ranks), otherwise ranks compile
  different graphs and their collectives deadlock.
