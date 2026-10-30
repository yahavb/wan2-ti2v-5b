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
| + upsample shortcut and its add in one compiled graph (`VAE_FUSE_SHORTCUT=1`) | 0.60 | 6.0s | 2.6s | 8.7s | 0.0487* |
| + 2 latent frames per VAE decoder call (`VAE_TCHUNK=2`) | 0.60 | 6.0s | 2.5s | 8.6s | 0.0487* |
| + self-attention output by all-to-all into an unsharded O-projection (`FAST_A2A_O=1`), FFN as one nkilib mlp kernel (`FAST_NKI_MLP=1`), modulation as 6 one-hot matmuls (`FAST_MOD_SPLIT=1`) | 0.58 | 5.8s | 2.5s | 8.4s | 0.0484* |
| + VAE in exact 8x1 strips with a one-row halo exchange per conv, instead of overlap-4 tiles (`VAE_STRIPS=1`) | 0.58 | 5.8s | 2.0s | 7.9s | 0.0484* |
| + the strips' norm + halo + cache step as one NKI kernel (`VAE_STRIP_NKI_NORM=1`) | 0.58 | 5.8s | 1.8s | 7.6s | 0.0484* |
| + T5 freed before the other models load; 4 latent frames per VAE decoder call (`VAE_TCHUNK=4`) | 0.58 | 5.8s | 1.7s | 7.5s | 0.0484* |
| + cross-attention over the real context tokens plus weighted copies of the identical pad key (`FAST_CROSS_MERGE=1`); VAE: sub-pixel interleave + shortcut as one NKI kernel (`VAE_SUBPIXEL_MERGE=1`), compiled own-rows middle attention (`VAE_STRIP_ATTN=1`), 1x1 shortcuts as one matmul (`VAE_CONV1X1_MM=1`) | 0.57 | 5.7s | 1.6s | 7.3s | 0.0485* |
| + VAE latent level in 2 calls instead of 6 (`VAE_LAT_ALL=1`), chunk copy-out behind the queued graphs (`VAE_SHIP_THREAD=1`), uint8 conversion on every rank (`VAE_U8_SHIP=1`) | **0.57** | **5.7s** | **1.5s** | **7.2s** | 0.0485* |

Env for the last row: `CFG_PARALLEL=1 CFG_GROUPS=interleave FAST_SP=1 FAST_LOCAL_CROSS=1
FAST_ATTN_IMPL=cte FAST_CROSS_IMPL=cte VAE_FUSE_NORM=1 VAE_STREAM=1 NKI_CONV3D_SMALL_COUT=1
VAE_SUBPIXEL=1 VAE_FUSE_CAT=1 VAE_FUSE_RES=1 VAE_FUSE_SHORTCUT=1 VAE_TCHUNK=4 DEVICE_SCHEDULER=1 MP4_VFORK=1
FAST_A2A_O=1 FAST_NKI_MLP=1 FAST_MOD_SPLIT=1 VAE_STRIPS=1 VAE_STRIP_NKI_NORM=1 FAST_CROSS_MERGE=1
VAE_SUBPIXEL_MERGE=1 VAE_STRIP_ATTN=1 VAE_CONV1X1_MM=1 VAE_LAT_ALL=1 VAE_SHIP_THREAD=1 VAE_U8_SHIP=1`. The `VAE_FUSE_CAT` and
`VAE_FUSE_RES` rows round to the same tenths as the row above them; run-to-run (DONE to DONE) they
are 8.97, 8.90 and 8.86 s, and the slowest VAE tile 2.71-2.73, 2.62-2.67 and 2.57-2.58 s.
`VAE_FUSE_SHORTCUT` writes a byte-identical mp4: it runs DupUp3D's repeat_interleave + permute and
the add as one graph and drops a clone of the stage input that nothing writes to. One 15x21 tile
on one core 2.298 -> 2.238 s; in the pipeline, 8 warm runs each with the two devices swapped,
VAE 2.659 -> 2.610 s and total 8.702 -> 8.657 s (both round to 8.7).
`VAE_TCHUNK=2` decodes latent frames 1-20 two per decoder call (frame 0 alone, as before), so every
kernel call does twice the work and the 1024-channel convs at the latent resolution, which saw one
frame per call, now take the nkilib kernel too. Four runs alternating the setting on one device
(4 warm runs each): VAE 2.59-2.62 -> 2.47-2.54 s, total 8.59-8.69 -> 8.48-8.63 s. One tile on one
core 2.15 -> 2.01 s. The tiled decode scores 44.71 dB against the exact fp32 decode (44.69 dB with
one frame per call) and 56.1 dB against the one-frame decode. Device memory at the VAE peak goes
from ~8.4 to ~10.6 GB per core (neuron-monitor), within the 12 GB per core; 4 frames per call
(11-18% faster per tile earlier) still runs out next to the resident DiT. Run 1 of a process
traces the new kernel shapes, so its VAE takes ~80 s instead of ~7 s; run 2 is warm.
Run 2 is warm. The nkilib conv filters are now packed when the model loads
(`Decoder3d.prepack_nki_filters`). Before, each conv after the first temporal upsample packed them
on its first call with 2+ output frames, in chunk 1 of the first decode. Chunk 0 of that decode
(1 frame) therefore ran the eager conv plus a separate residual add, and run 2 was the first to
trace the kernel at the 1-frame shapes: its VAE took 11-20 s instead of 2.6 s, which is what
`NUM_RUNS=2` reported. Run 1's frames also differed from every later run's by one bf16 rounding in
chunk 0 under `VAE_FUSE_RES` (rel-RMS 1.7e-3). Now run 2's VAE takes 2.5-2.6 s, every run writes
the same mp4, and the steady-state output is unchanged (bit-identical decode with the loop flags on
and off).

The last row's three DiT changes, per block on the block bench (`scripts/validate/block_bench.py`,
`CFG_PAR=1`, interleaved cores, 30 reps; 19.71 ms before): `FAST_A2A_O` 19.45 ms, `FAST_NKI_MLP`
19.44 ms, both 19.03-19.11 ms, all three 18.82-18.84 ms; block output rel-RMS vs the fp32 block
0.0046 in every case, as before. `FAST_A2A_O` turns the O-projection's reduce-scatter (36 MB
per rank) into one all-to-all of the attention output (9 MB) and keeps an unsharded copy of each
O weight (+0.43 GB per rank). `FAST_NKI_MLP` replaces the FFN's two matmuls and the GELU-tanh, which
the compiler decomposes into ~5 ScalarE passes, with one nkilib `mlp` call. `FAST_MOD_SPLIT` drops
the [tokens, 6, C] modulation tensor (72 MB per block). In the pipeline, alternating runs on one
device (4 warm runs each): denoise 6.03-6.10 -> 5.85-5.94 s with the first two, total 8.53-8.62 ->
8.38-8.47 s; adding `FAST_MOD_SPLIT`, denoise 5.85-5.96 -> 5.81-5.90 s, total 8.35-8.49 ->
8.34-8.45 s. The output is not bit-identical (different accumulation in the mlp kernel and the
modulation products); the step-0 noise prediction scores 0.0484 vs fp32 (0.0487 before), and
every run of a process writes the same mp4.

`VAE_STRIPS=1` decodes the frame as 8 exact horizontal strips, one per rank, instead of 8
overlap-4 tiles of 15x21 latent pixels that together cover 1.68x the frame. Up to the first
upsample each rank runs a 6-row latent window (neighbouring windows share a row, since 44 rows
do not split into 8), then crops to its exact 11 rows at 2x; before every spatial conv each rank
gets the row above and the row below from its neighbours (zeros at the image border), and the
middle attention block runs on the all-gathered full frame. So the strips compute what an untiled
decode computes, up to bf16 rounding. The exchange is three graphs per conv: the strip's edge rows,
an all-gather alone in its graph, then the norm + cache-concat graph with the neighbours' rows
sliced in. Inside the norm graph the same all-gather cost ~10x more (3.3 ms instead of 0.36 ms
at the 8x level, `.remote/halo2_bench.py`), which is what made the in-graph version a wash before.
Same final latent, all 8 ranks, no DiT resident (`scripts/validate/vae_tiled_check.py`): decode
2.50 -> 1.96 s, and against the exact fp32 decode 44.71 -> 52.96 dB (worst frame 42.93 -> 51.94
dB). In the pipeline, alternating runs on one device (3 warm runs each): VAE 2.50-2.54 -> 2.02-2.08 s,
total 8.32-8.40 -> 7.78-7.93 s; device memory during the VAE ~9.9 -> ~9.5 GB per core (neuron-monitor).
Every run writes the same mp4. Run 1 of a process traces the strip kernel shapes (VAE ~145 s).

`VAE_STRIP_NKI_NORM=1`: rank 0's system trace of the strip decode (`.remote/vae_seq.py` lists one
decoder call's executions in order) put the norm + halo + cache-concat graph first: 3.21 ms per call
for C=256 and 7.8 ms for C=512 at the 8x level, against 1.3 ms for the same norm without halo rows,
27 ms of the call's 170 ms. `kernels/vae_strip_norm.py` streams each frame through SBUF in
2048-pixel blocks with channels on the partitions, sums the squares over channels with a
ones-matrix matmul (which also broadcasts the sum to every partition) and writes the normed rows,
the neighbours' halo rows and the cached frames straight into the conv input and the next chunk's
cache. One core (`scripts/validate/strip_norm_check.py`), compiled graph -> kernel: 1.88 -> 0.58 ms
(C=256, 8 frames, 44x272), 3.62 -> 1.15 ms (C=512), 0.63 -> 0.34 ms (C=512, 22x136), equal at the
two lowest levels; values within one bf16 step of the graph's (rel-RMS <= 2.7e-3), halo rows and
cached frames identical. The gathered edge rows are now laid out one contiguous block per row, as
the kernel reads them. NKI's cross-process trace cache keys a kernel by its qualified name and
argument shapes, not closure values, so each (shape, halo source) kernel gets its own name: with one
name, ranks with equal shapes and other halo sources reused each other's traces (32 dB). Same
latent, 8 ranks: decode 1.94 -> 1.74 s, 52.95 -> 53.01 dB vs the exact fp32 decode. Pipeline,
alternating runs (2 warm runs each): VAE 2.01-2.03 -> 1.78-1.81 s, total 7.83-7.84 -> 7.56-7.61 s.

T5 is now freed before the VAE and DiT weights are moved to its rank: it encodes both prompts
right after it loads, the outputs go to the CPU, and its ~11 GB are freed before anything else
lands on the device. Freed after them (as before), T5 left ~1.9 GB of cached fragments on its
rank for the whole run (reserved after the VAE 9.9 GB vs 7.8 GB on the other ranks; returning the
cache before the VAE did not help, the fragments re-formed), and its HBM-stack neighbour ran out
of memory in the VAE with 4 latent frames per call. Latent and mp4 are byte-identical to before.
With the strips, `VAE_TCHUNK=4` now fits: alternating runs on one device, VAE 1.79-1.80 ->
1.67-1.70 s and total 7.56 -> 7.44-7.52 s over 5 processes without an allocation failure; device
memory at the VAE peak 10.55-10.75 GB per core (neuron-monitor; the two cores of an HBM stack
together at most 21.5 of 24 GB). 53.00 dB vs the exact fp32 decode (53.01 with 2 frames per call),
57.3 dB vs the 2-frame decode; one mp4 hash over 4 runs. Most of the gain is per-call overhead:
6 decoder calls instead of 11, so half the halo all-gathers and graph launches.

`FAST_CROSS_MERGE=1`: the T5 context is zero-padded to 512 tokens and the model has no context mask,
so all padded positions have the same text embedding and therefore the same K and V in every block.
Attention over m identical keys equals attention over c copies of that key with log(m / c) added to
their scores, so the 512 keys become the n real ones (70 for the prompt, 126 for the negative
prompt) plus copies of the pad key, 128 in all for both CFG groups; exact up to rounding (1e-8
rel-RMS in fp64, `scripts/validate/cross_merge_bench.py`). The cross-attention then runs as SDPA
with the log-weights as an fp32 additive mask: 0.39 ms per block against 1.11 ms for
`attention_cte` on 512 keys (attention_cte itself got slower at 128 keys, 2.6 ms). The premise is
checked on the context at run time (its nonzero rows must be a prefix), else all 512 keys are used.
VAE (one 4-frame decoder call in rank 0's trace: 268 graph executions, 274 ms busy): the end of each
upsample stage was an eager stack + permute of the 4 sub-pixel phases and the compiled shortcut +
add graph (~4-8 ms per call at the 2x->4x and 4x->8x stages); `kernels/vae_subpixel_merge.py` does
the interleave and the DupUp3D shortcut add in one pass, bit-identical (4x->8x stage, 16 frames, one
core: 3.87 -> 1.20 ms; `scripts/validate/subpix_merge_check.py`). The middle attention block ran as
~90 eager executions per call (a per-frame NKI loop on the gathered full frame); `VAE_STRIP_ATTN`
computes only the rank's own rows against the gathered frame in one compiled graph (fp32 scores
and softmax; 57.2 dB vs the NKI-loop decode). The two 1x1x1 ResidualBlock shortcuts ran as an eager
zero `F.pad` copy plus the compiler's conv; `VAE_CONV1X1_MM` makes them one matmul graph,
bit-identical. Same latent, 8 ranks: decode 1.60 -> 1.54 s with the first and third, 53.00 dB vs the
exact fp32 decode. Pipeline, alternating runs on one device with everything in this row (2 warm runs
each, the dump run excluded): denoise 5.76-5.84 -> 5.65-5.74 s, VAE 1.675-1.678 -> 1.61-1.62 s
(decode 1.52 -> 1.45-1.46 s), total 7.43-7.51 -> 7.27-7.36 s; step-0 noise prediction 0.0485 vs fp32;
VAE-phase device memory 10.2-10.4 GB per core (was 10.6-10.8); one mp4 hash over 4 runs.

The latent-resolution part of the decoder (conv1, the middle blocks and the first up block, ~30 ms
of every 4-frame call) runs per chunk only because the rest of the decoder does. `VAE_LAT_ALL`
runs it in two calls, latent frame 0 and then the other 20 frames, before the chunked rest: the
causal convs see the same frames and the first up block's temporal upsample starts on frame 1 with
zero padding either way, so the frames are bit-identical (decode 1.40 -> 1.35 s on the same
latent). Each rank's copy-out of a chunk was a `.cpu()` that waits for everything queued, i.e.
for the whole next chunk, then a file write with nothing queued behind it (~5 ms of device idle
per decoder call in the trace); `VAE_SHIP_THREAD` queues a `non_blocking` copy right behind the
chunk's graphs with an event after it, and a helper thread waits on the event and writes the file
(`scripts/validate/d2h_bench.py`: the copy is stream-ordered and returns at once). `VAE_U8_SHIP` has every
rank convert its own rows to uint8 before shipping them, so rank 0 only stacks uint8 rows; the
frames are no longer concatenated into one 93 MB array inside the timed phase. Pipeline,
alternating runs on one device (2 warm runs each, the dump run excluded): VAE 1.607-1.622 ->
1.489-1.493 s (one 1.604 in the run with the memory monitor), total 7.29-7.35 -> 7.16-7.20 s; the
latent and the mp4 bytes are identical to the configuration above (same md5 in every run), and the
VAE-phase device memory drops to 9.2-9.4 GB per core (neuron-monitor peak).

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
frames per decoder call was 11-18% faster per tile on an empty core (`VAE_TCHUNK`; 2 frames is now on, see above) but
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
