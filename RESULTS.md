# Benchmark log

All runs: one trn2 NeuronDevice, 10 denoising steps, CFG on, same prompt/image/seed.
Numbers are warm runs (after compile). Measured 2026-10-10 to 2026-10-12.

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
