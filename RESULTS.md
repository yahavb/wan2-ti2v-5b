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
| + CFG groups on alternating cores (`CFG_GROUPS=interleave`) | 0.62 | 6.2s | 3.5s | 9.6s | 0.0487* |
| + streamed VAE output: blend + mp4 encode per latent frame during the decode (`VAE_STREAM=1`) | 0.62 | 6.2s | 3.0s | 9.2s | 0.0487* |
| + VAE head conv (256->12) on nkilib (`NKI_CONV3D_SMALL_COUT=1`), sub-pixel upsample convs (`VAE_SUBPIXEL=1`) | 0.61 | 6.1s | 2.8s | 8.9s | 0.0487* |
| + norm and causal-cache concat in one compiled graph (`VAE_FUSE_CAT=1`) | 0.61 | 6.1s | 2.7s | 8.8s | 0.0487* |
| + ResidualBlock shortcut added inside the nkilib conv (`VAE_FUSE_RES=1`) | 0.61 | 6.1s | 2.7s | 8.8s | 0.0487* |
| + UniPC step on the device (`DEVICE_SCHEDULER=1`), mp4 encoder spawned with vfork (`MP4_VFORK=1`) | 0.60 | 6.0s | 2.6s | 8.7s | 0.0487* |
| + upsample shortcut and its add in one compiled graph (`VAE_FUSE_SHORTCUT=1`) | **0.60** | **6.0s** | **2.6s** | **8.7s** | 0.0487* |

Env for the last row: `CFG_PARALLEL=1 CFG_GROUPS=interleave FAST_SP=1 FAST_LOCAL_CROSS=1
FAST_ATTN_IMPL=cte FAST_CROSS_IMPL=cte VAE_FUSE_NORM=1 VAE_STREAM=1 NKI_CONV3D_SMALL_COUT=1
VAE_SUBPIXEL=1 VAE_FUSE_CAT=1 VAE_FUSE_RES=1 VAE_FUSE_SHORTCUT=1 DEVICE_SCHEDULER=1 MP4_VFORK=1`. The `VAE_FUSE_CAT` and
`VAE_FUSE_RES` rows round to the same tenths as the row above them; run-to-run (DONE to DONE) they
are 8.97, 8.90 and 8.86 s, and the slowest VAE tile 2.71-2.73, 2.62-2.67 and 2.57-2.58 s.
`VAE_FUSE_SHORTCUT` writes a byte-identical mp4: it runs DupUp3D's repeat_interleave + permute and
the add as one graph and drops a clone of the stage input that nothing writes to. One 15x21 tile
on one core 2.298 -> 2.238 s; in the pipeline, 8 warm runs each with the two devices swapped,
VAE 2.659 -> 2.610 s and total 8.702 -> 8.657 s (both round to 8.7).
Run 2 is warm. The nkilib conv filters are now packed when the model loads
(`Decoder3d.prepack_nki_filters`). Before, each conv after the first temporal upsample packed them
on its first call with 2+ output frames, in chunk 1 of the first decode. Chunk 0 of that decode
(1 frame) therefore ran the eager conv plus a separate residual add, and run 2 was the first to
trace the kernel at the 1-frame shapes: its VAE took 11-20 s instead of 2.6 s, which is what
`NUM_RUNS=2` reported. Run 1's frames also differed from every later run's by one bf16 rounding in
chunk 0 under `VAE_FUSE_RES` (rel-RMS 1.7e-3). Now run 2's VAE takes 2.5-2.6 s, every run writes
the same mp4, and the steady-state output is unchanged (bit-identical decode with the loop flags on
and off).

Host stalls, from the runtime system trace (`scripts/prof/sys_phases.py`). Each denoising step left
the device idle ~19 ms: ~7 ms for the CPU UniPC round trip after the head, then ~12 ms in the next
step's first in-graph collective, waiting for the ranks whose single-threaded CPU scheduler was
slower. `DEVICE_SCHEDULER=1` (`models/device_unipc.py`) runs the scheduler's linear combinations on
the device, with its scalar coefficients computed on the CPU as upstream does, applied with the same
PyTorch type-promotion rounding: bit-identical to the CPU scheduler on CPU and on device
(`scripts/validate/unipc_device_check.py`), and the pipeline's mp4 is byte-identical. Denoise device
busy 96.9% -> 99.5%, idle 0.19 -> 0.03 s. In the VAE, starting ffmpeg with imageio's `preexec_fn`
forks this tens-of-GB process; the page-table copy blocked the runtime's completion thread for
88-95 ms once per decode (a `kbl_exec_post` of that length in the trace). `MP4_VFORK=1` lets Python
spawn it with vfork: ~1 ms instead of 14-20 ms for a 20 GB process measured in isolation, and the
stall is gone (VAE idle 0.375 -> 0.265 s). Six warm runs, same build:
both flags 8.59-8.72 s total, `DEVICE_SCHEDULER` alone 8.70-8.87 s, neither 8.81-9.18 s. Tried and
dropped: more CPU threads on ranks 1-7 (no clear change), freezing/disabling Python GC during a run
(no collection over 5 ms happens in a warm run), copying VAE frames to the host from a helper thread
(longer tail: the copies queue behind the device work).
\* The fused norm also runs in the VAE encoder, so the conditioning latent (and the step-0 input)
changes slightly; these rows are scored against an fp32 reference computed on their own input.
`CFG_GROUPS=interleave` gives bit-identical output to the contiguous layout. `VAE_STREAM=1` writes a
byte-identical mp4: the decode, blend and uint8 conversion do the same per-element math, only
chunk by chunk; what it removes is the 0.53 s that ran after the decode (collect+sum 0.16 s,
convert 0.03 s, mp4 write 0.34 s), leaving a 0.07-0.12 s tail.

VAE conv changes, one 15x21 tile on one core (`scripts/validate/vae_bench.py`, `UNTIMED_REPS=3`; one latent frame per decoder call, as in the
pipeline; four jobs on one device): 2.72 s -> 2.61-2.67 s with the head conv on nkilib (eager was
the only non-nkilib 3x3x3 conv at full tile resolution; rel-RMS 2.3e-5 vs eager) -> 2.47-2.53 s
adding sub-pixel upsample convs -> 2.35 s adding `VAE_FUSE_CAT`. Sub-pixel: nearest-2x followed by a
3x3 conv equals four 2x2 convs on the low-res input (one per output phase, taps summed), so 2.25x
fewer MACs and no 4x upsampled tensor; the summed taps are rounded to bf16 once, which moves the
output by rel-RMS 3.0e-3 (58.3 dB vs the previous bf16 decode) and leaves the error against the
exact fp32 decode at 44.68 dB (was 44.69). `VAE_FUSE_CAT` gives bit-identical frames.
`VAE_FUSE_RES` (2.35 -> 2.30 s) rounds conv + shortcut to bf16 once instead of twice: 44.69 dB vs
the exact decode. Tried and dropped: nkilib for the D_out=1 convs at latent resolution (2.46-2.50
vs 2.47 s).

Exact (seam-free) tiling, measured but not built: a non-overlapping 6x34 strip decodes in 1.65 s
on one core and an 11x17 tile in 1.62 s, vs 2.35 s for today's overlap-4 15x21 tile. Exactness
needs a 1-px halo exchange before every 3x3 conv (~33 per latent frame, ~690 per decode) plus a K/V
all-gather for the middle attention. In-graph exchange fused into the norm graph
(`scripts/validate/halo_bench.py`, 8 ranks; `MODE=strips` for the strips) costs +0.34/+0.44/+0.75/+3.30 ms per conv for 4x2 tiles at
1x/2x/4x/8x resolution and +0.24/+0.54/+1.91/+1.76 ms for 8x1 strips: ~0.65 s per decode, which
cancels the 0.7 s of saved compute. Wider halos exchanged every few convs could net ~0.3 s.

VAE accuracy, same final latent, against an exact fp32 CPU decode without tiling
(`scripts/validate/vae_cpu_ref.py`): tiled bf16 decode (overlap 4) 44.67 dB, with
`VAE_FUSE_NORM=1` 44.69 dB; fused vs unfused 55.75 dB. Overlap 3: 43.74 dB (3.01 s), overlap 2:
42.24 dB (2.54 s) vs 3.10 s for overlap 4 -- not taken, the speed costs 1-2.5 dB.

Per-call attention microbench (one core, 6 heads x 7854 tokens, d=128, layer-25 QK-norm gammas):
sdpa 18.2 ms, const-max 5.8 ms, `attention_cte` 5.3 ms, all at bf16-level error (rel-RMS 0.0055
vs fp32). Per-block (TP4, batch 1): 29.4 ms (TP8 batch 2) -> 24.5 (CFG-parallel) -> 21.9 (+SP) -> 20.6 ms (+local cross).

Interleaving the CFG groups gives 19.8 ms per block; where that goes is in the profile section
below. An earlier figure here (collectives 6.2 of the contiguous block's 20.6 ms) was wrong: that
probe replaced each reduce-scatter with a row slice, which let the compiler skip three quarters of
the O-projection and FFN-down rows (the 2.3 ms difference matches those matmuls). With a stand-in
that reads every row, the collectives cost 3.1 ms in place (4.0 ms standalone). Per denoising step
the host side (D2H, CPU UniPC step, H2D) takes 4-6 ms (`STEP_TIMING=1`); the device sits idle
~19 ms per step (system trace), which also covers the host work before the first block.

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

## Where the 8.8 s goes (`loop_engineering`, profiled 2026-10-28)

One warm run of the last configuration above, 8.81 s: denoise 6.13 s (device busy 96.9% on
rank 0) and VAE 2.68 s (device busy 86.0%). Headroom means savings that keep the output the same
apart from bf16 rounding.

| Area | Time now | Share | Running at | Floor | Realistic headroom | How |
|---|---|---|---|---|---|---|
| DiT linear layers | 2.47 s | 28% | 83% of peak | 2.05 s | ~0.2 s | Close to peak already; the FLOPs are fixed by the model at bf16 |
| DiT attention (self + cross) | 1.77 s | 20% | 45% of peak (self), 17% (cross) | 0.73 s | 0.4-0.6 s | A faster flash-attention kernel, around 60-70% of peak; cross-attention has only 512 keys, which suits the kernel poorly |
| VAE convolutions | 1.74 s | 20% | 79% on the main convs, 30% at 15x21 | 1.30 s (0.77 s with exact tiles) | 0.3-0.6 s | Overlapping tiles do 1.68x the needed work; exact tiles need a halo exchange cheaper than the 0.65 s measured. 4 frames per call runs at 88-94% of peak but runs out of HBM next to the resident DiT |
| DiT collectives | 0.94-1.2 s | 11-14% | mostly serialized with compute | ~0 if hidden | 0.2-0.5 s | Overlap them with the matmuls. All-to-all moves fewer bytes (1.0 vs 4.0 ms per block) but needs the full weights on every core, which doesn't fit next to the VAE |
| DiT pointwise (norms, modulation, gates, RoPE, QK-norm) | 0.76 s | 9% | memory / vector bound | small if fused | 0.3-0.5 s | Fuse norm + modulation into the QKV and FFN-up matmuls, gate + residual into O-proj and FFN-down, RoPE + QK-norm into attention |
| VAE norms, copies, small graphs | 0.56 s | 6% | memory bound (norms at ~130 GB/s) | small if fused | 0.1-0.3 s | Fold the shortcut and interleave copies into the conv kernels, so fewer graphs run |
| Host / idle time | 0.57 s | 6% | device waiting on the host | 0 | 0.25-0.45 s | Async per-frame VAE copies; find the 0.09 s VAE stall; UniPC step on the device (19 ms idle per step); fewer VAE graph launches (3,540 per tile) |
| DiT head, embeddings | 0.04 s | <1% | | | ~0 | |
| **Total** | **8.81 s** | | | **~3.6-4.1 s** | **~2-3 s** | ~6-7 s end to end if most of these land |

Floor = the same work at 100% of the bf16 peak (83.4 TFLOP/s per core) with everything else
hidden: a lower bound, not a target. The attention floor counts only the matmuls. The fusion
headroom (DiT pointwise, VAE small graphs) is an estimate, and the ranges overlap, so they don't
add exactly. The 88-94% for 4 frames per call is the nkilib conv on 4-frame shapes, measured per
conv in the parallel conv-profiling session. Not counted, because they change the output: FP8
(torch's `float8_e4m3fn` is rejected by neuronx-cc 2.28 on trn2; NKI's own fp8 types were not
tried), fewer steps, step caching.

The same profile, by bottleneck:

| # | Bottleneck | Time per video | Share | Detail | Source |
|---|---|---|---|---|---|
| 1 | DiT linear layers (QKV, O-proj, FFN up/down, cross-attn q/o) | 2.47 s | 28% | 8.25 ms per block; 83% of bf16 peak (QKV alone 54%) | bench |
| 2 | DiT self-attention (`attention_cte`) | 1.51 s | 17% | 5.0 ms per block; 45% of peak | ablation |
| 3 | VAE 3x3x3 convs on nkilib (30x42 to 120x168) | 1.48 s | 17% | 8 graphs, 2.5-8.6 ms per call; 69-84% of peak | trace |
| 4 | DiT collectives (2 all-gathers + 2 reduce-scatters per block) | 0.94-1.2 s | 11-14% | 3.1 ms per block in place, 4.0 ms standalone | ablation, bench |
| 5 | DiT elementwise (norms, modulation, gates, residual adds) | ~0.43 s | ~5% | ~1.4 ms per block | derived |
| 6 | VAE device idle | 0.38 s | 4% | gaps between 3,540 eager graph launches 0.15 s; wait at the end of the decode (last-frame assembly, slower tiles) 0.10 s; one host stall 0.09 s (cause not found); blocking per-frame copies to host 0.04 s | trace |
| 7 | VAE other small graphs (~170) | 0.33 s | 4% | shortcut and interleave copies, 1x1 and time convs, head conv, attention; not split further | trace |
| 8 | DiT cross-attention | 0.26 s | 3% | 0.9 ms per block | ablation |
| 9 | VAE RMS_norm + SiLU (+ cache concat) graphs | 0.23 s | 3% | ~130 GB/s | trace |
| 10 | DiT QK-norm | 0.19 s | 2% | 0.6 ms per block | ablation |
| 11 | DiT device idle | 0.19 s | 2% | ~19 ms per step: UniPC step on the CPU plus copies after the head, and host work before the first block | trace |
| 12 | VAE eager 3x3x3 convs at 15x21 (1 frame per call) | 0.15 s | 2% | 0.72 ms per call; 30% of peak | trace |
| 13 | DiT RoPE | 0.14 s | 2% | 0.5 ms per block | ablation |
| 14 | VAE sub-pixel upsample convs | 0.11 s | 1% | 12 graphs | trace |
| 15 | DiT head (CFG combine), patch and time embedding | 0.04 s | <1% | head 3.3 ms per step | trace |

Sources:
- trace: rank 0's runtime system trace of one full pipeline run (`NUM_RUNS=3`; the pipeline run
  under `scripts/prof/rank_env.sh` with `SYSTRACE_DIR=<dir> SYSTEM_RANKS=0 PROFILE_RANKS=none`),
  exported with `scripts/prof/sys_to_parquet.sh` and split with
  `scripts/prof/sys_phases.py <parquet> 3 300`. The trace names the VAE graphs only by hash; they
  were identified by calls per decode and time per call, against the call sites and counts from
  `scripts/validate/vae_op_profile.py` and the convs' FLOP counts. VAE rows are rank 0's tile.
- ablation: `scripts/validate/block_bench.py sp_lx` with `CFG_PAR=1` and one of `SP_NOCOMM`,
  `SP_NOQKN`, `SP_NOROPE`, `SP_NOSA`, `SP_NOXA` set to 1, ranks on interleaved cores
  (`NEURON_RT_VISIBLE_CORES=c,c+2,c+4,c+6,c+1,c+3,c+5,c+7`), compared with the full block
  (19.80 ms). With all five removed the block takes 10.0 ms.
- bench: standalone graphs at the block's shapes: `scripts/validate/block_mm_bench.py` (the 8
  matmuls in one graph, 8.25 ms) and `scripts/validate/sp_cc_bench.py` (per call: all-gather
  0.83 ms, reduce-scatter 1.18 ms; the 4 all-to-alls of a block 1.01 ms).
- derived: the block's in-place remainder after the five removals (9.7 ms) minus the matmul bench.

DiT rows are per block x 30 blocks x 10 steps; the two CFG groups run at the same time, so time on
one rank equals wall time.

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
