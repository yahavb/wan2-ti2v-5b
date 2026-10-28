# Faster than an A100: Wan2.2-TI2V-5B on one Trainium2 device

re:Invent code talk: talking points and slide plan. Every number comes from
[RESULTS.md](../../RESULTS.md). Code links are pinned to commit `788674b`.

Spec for every run: 704x544, 81 frames, 10 denoising steps, CFG on, the same prompt, image
and seed, one trn2 NeuronDevice exposed as 8 logical cores (LNC1). Times are warm runs.

`GH` = `https://github.com/yahavb/wan2-ti2v-5b/blob/788674b`

---

## Slide 1. The goal

**Talking point:** one video, end to end, faster than an A100.

| Platform | Denoise | VAE + output | Total |
|---|---|---|---|
| A100, torch.compile, BF16 VAE + autotuning | 11.43 s | 5.48 s | **16.92 s** |
| Trainium2, where we started (original eager DiT) | 44.3 s | 32.3 s | 76.6 s |
| Trainium2, today (`main`) | 6.0 s | 2.6 s | **8.7 s** |

- 76.6 s to 8.7 s is 8.8x on the same hardware. Against the A100 it is 1.9x faster.
- The validation run on 2026-10-28 measured 7.1 s (2.4x). **Check before using it:** that job
  may have run on a trn3-dev1 node. In the same job, main took 11.8 s, while RESULTS.md has main
  at 16.2 s on trn2. Until the node is confirmed, quote 8.7 s.

**Animation:** a horizontal bar per milestone that shrinks from 76.6 s down to 8.7 s, with the
A100 drawn as a fixed red line at 16.92 s. The bar crosses the line at `08921e3` (16.1 s).

Full ladder for the animation (from RESULTS.md):

| Step | Change | Denoise | VAE | Total |
|---|---|---|---|---|
| 0 | original eager DiT | 44.3 | 32.3 | 76.6 |
| 1 | `ce390d3` block-compiled DiT, batched CFG (TP=8) | 19.1 | 17.7 | 36.8 |
| 2 | `2e4a6a6` cross-attn K/V cache + tiled VAE | 19.2 | 9.0 | 28.2 |
| 3 | `b8d5cbd` const-max self-attention on 22/30 layers | 11.9 | 9.0 | 21.0 |
| 4 | `de16d5d` tiles weighted on device, collected via /dev/shm | 11.9 | 6.0 | 17.9 |
| 5 | `679701a` nkilib conv3d for the VAE | 11.9 | 5.4 | 17.3 |
| 6 | `99f92da` upsample convs on nkilib | 11.9 | 5.2 | 17.2 |
| 7 | `08921e3` rank-0 CPU threads 16, in-place uint8 convert | 11.9 | 4.2 | **16.1** |
| 8 | `attention_cte` self + cross | 9.0 | 4.1 | 13.2 |
| 9 | CFG-parallel: 2 groups x TP4 | 7.6 | 4.3 | 11.9 |
| 10 | Megatron sequence parallelism | 6.8 | 4.4 | 11.1 |
| 11 | cross-attention local to the token shard | 6.4 | 4.4 | 10.8 |
| 12 | fused VAE RMS_norm + SiLU | 6.4 | 3.5 | 10.0 |
| 13 | interleaved CFG groups | 6.2 | 3.5 | 9.6 |
| 14 | streamed VAE output + mp4 | 6.2 | 3.0 | 9.2 |
| 15 | sub-pixel upsample, head conv on nkilib, fused cat, fused residual | 6.1 | 2.7 | 8.8 |
| 16 | UniPC on the device, vfork the encoder, fused shortcut | 6.0 | 2.6 | **8.7** |

---

## Slide 2. The model in one picture

- T5 text encoder (one rank), a 30-block DiT (dim 3072, 24 heads, 7,854 video tokens), and a
  causal 3D VAE decoder.
- The DiT runs 30 blocks x 10 steps x 2 CFG passes = 600 block calls per video.
- The VAE decodes 21 latent frames into 81 frames at 704x544.

**Talking point:** the two halves need opposite strategies. The DiT is a big, repeated, regular
graph, so let the compiler see more of it. The VAE is a long chain of convolutions on changing
shapes, and compiling it made things worse. Hand-written kernels fixed it.

---

## Slide 3. Change what you compile: the DiT (76.6 s to 36.8 s)

**Before:** eager DiT. Only pieces such as the FFN were compiled. Every TP all-reduce forced a
graph break on purpose, so each block ran as many small graphs with Python in between.

[`GH/models/tp_utils.py#L119-L128`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/tp_utils.py#L119-L128)

```python
@torch.compiler.disable
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """... force a graph break: the compiled FFN NEFF handles only matmuls,
    then all-reduce runs in eager."""
```

**After (`DIT_FAST=1`):** one compiled graph per transformer block, all-reduces inside the graph
as functional collectives, and both CFG passes in one batch of 2.

[`GH/models/fast_dit.py#L90-L100`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L90-L100)

```python
def _all_reduce(x):
    return funcol.all_reduce(x, "sum", get_tp_group())   # traced: stays inside the NEFF
```

[`GH/models/fast_dit.py#L260-L311`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L260-L311)
is the whole block as a single graph: norm + modulation, QKV, QK-norm, RoPE, attention,
O-proj + all-reduce, cross-attention, FFN + all-reduce, gated residuals.

```python
kw = dict(backend="neuron", dynamic=False, fullgraph=FAST_FULLGRAPH)
self.blocks = [torch.compile(_FastBlock(b, ...), **kw) for b in model.blocks]
```

What the bigger scope buys:
1. One NEFF launch per block instead of many per-op launches with host gaps between them.
2. Fusion of norms, modulation, RoPE and residual adds with their neighbours.
3. The compiler schedules each collective next to the matmul that produces it.
4. All 30 blocks share one NEFF, because the weights are graph inputs. One compile serves 600 calls.

Numbers: 4.4 s/step to 1.91 s/step (2.3x) on denoise.

**Animation:** two system-trace style timelines for one block. Before: many short boxes with
gaps between them. After: one long box.

---

## Slide 4. Change what you compile: the VAE (the opposite lesson)

**Talking point:** bigger compile scope is not always better.

| Attempt | VAE time | Compile | Outcome |
|---|---|---|---|
| `torch.compile` of the original decoder | 11.5-12.9 s | 58 min | dropped |
| Static-cache decoder, 2 graphs (vllm-omni port) | 12.8-14.2 s | 20.6 min | dropped |
| `torch.compile` across the whole VAE (early attempt) | 400+ NEFFs | — | out of memory |
| **Eager + NKI kernels, tiled over 8 ranks** | **1.7 s** | none | kept |

(Measured at 368x480x21, where the VAE was the bottleneck.)

Why: the compiler-generated 3D convolutions were slow, and the causal cache changes shapes
from chunk to chunk. Hand-written NKI kernels won: conv2d as shifted matmuls, single-head VAE
attention, then nkilib `conv3d` for the causal 3x3x3 convs.

[`GH/kernels/vae_conv3d.py#L33-L58`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L33-L58)

```python
@nki.jit
def _conv3d_3x3x3_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1))

_conv3d_3x3x3 = wrap_nki(_conv3d_3x3x3_kernel)   # callable from eager PyTorch
```

The rule that decides when to use it: C_in and C_out >= 96, and at least 2 output frames.
Below that the eager conv is faster. A conv with 1 output frame regressed the decode by 25%.

**Animation:** a compile-time bar (58 min, 20.6 min) next to a latency bar (12.9 s vs 1.7 s).

---

## Slide 5. Attention, part 1: const-max softmax (1.92 to 1.19 s/step)

**Idea:** flash attention keeps a running max of the scores so the exp never overflows. If you
know a bound on the scores ahead of time, you can subtract a constant instead and skip the
running-max bookkeeping.

**Catch:** it is only exact when the scores stay under the kernel's clip. On real activations,
using it on all layers broke the video after frame 0.

**Fix:** bound each layer's worst-case score from its QK-norm weights, and use const-max only
where the bound is <= 70. That came to 22 of 30 layers, with exact SDPA on the rest.

[`GH/models/fast_dit.py#L77-L84`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L77-L84)

```python
def const_max_layer_ok(block):
    gq = sa.norm_q.weight.abs().reshape(-1, d).amax(1)
    gk = sa.norm_k.weight.abs().reshape(-1, d).amax(1)
    hard = float((d ** 0.5) * (gq * gk).max())   # worst-case scaled score
    return hard <= CONST_MAX_HARD_LIMIT, hard
```

Two more lessons:
- All heads at once overflowed SBUF and crashed the compiler, so the kernel is called on as
  many heads at a time as fit:
  [`GH/models/fast_dit.py#L210-L225`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L210-L225).
- The layer choice is a max over ranks. If ranks pick different layers, they compile different
  graphs and their collectives deadlock.

Per call (6 heads x 7,854 tokens): SDPA 18.2 ms, const-max 5.8 ms.

---

## Slide 6. Attention, part 2: attention_cte, CFG-parallel, sequence parallel (16.2 to 10.8 s)

| Change | s/step | Total |
|---|---|---|
| main (const-max 22/30 + SDPA, TP=8, batch-2 CFG) | 1.20 | 16.2 s |
| nkilib `attention_cte` self-attention, all 30 layers | 0.94 | 13.7 s |
| + `attention_cte` cross-attention | 0.90 | 13.2 s |
| + CFG-parallel: 2 groups x TP4, batch 1 each | 0.76 | 11.9 s |
| + Megatron sequence parallelism | 0.68 | 11.1 s |
| + cross-attention local to the token shard | 0.64 | 10.8 s |

1. **`attention_cte`** is exact flash attention with no score bound. It is 5.3 ms per call, the
   same speed as const-max but safe on every layer.
   [`GH/models/fast_dit.py#L228-L241`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L228-L241)
   ```python
   from nkilib.core.attention.attention_cte import attention_cte
   out = attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False)
   ```
2. **CFG-parallel:** the conditional and unconditional passes run at the same time on two
   groups of 4 cores, instead of one batch of 2 on 8 cores. Smaller TP means fewer, cheaper
   collectives per matmul. `CFG_GROUPS=interleave` puts the groups on alternating cores (+0.4 s).
   [`GH/inference_neuron_ti2v.py#L241-L246`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L241-L246)
3. **Sequence parallelism:** each rank keeps only its token shard of the residual stream. An
   all-gather goes before each column-parallel matmul and a reduce-scatter after each
   row-parallel matmul. That moves the same bytes as the all-reduce, but the norms, modulation
   and residual adds run on 1/4 of the tokens.
   [`GH/models/fast_dit.py#L314-L390`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L314-L390)
   ```python
   hf = _ag(h)                                  # [S, C] -> [Np, C]
   q, k, v = _lin(hf, sa.q), _lin(hf, sa.k), _lin(hf, sa.v)
   ...
   y = _rs(_mm(y, sa.o)) + sa.o.bias            # [Np, C] -> [S, C]
   ```
4. **Local cross-attention:** cross-attention has only 512 text keys. Each rank runs it on its
   own token shard with all heads, so it needs no collectives.

Per block: 29.4 ms (TP8 batch 2) to 24.5 (CFG-parallel) to 21.9 (+SP) to 20.6 (+local cross)
to 19.8 ms (interleaved).

**Animation:** 8 cores as a grid. First all 8 do the batch of 2, then they split into two
colours (cond / uncond), then each core's token shard lights up for the pointwise ops.

---

## Slide 7. VAE: tiles across all 8 ranks (9.0 to 6.0 s)

- Before, the VAE ran on rank 0 while 7 cores sat idle. Now every rank decodes overlapping
  spatial tiles of the latent (overlap 4).
- Each rank multiplies its tile by a blend weight that is already normalized, on the device, and
  writes it to `/dev/shm`. Rank 0 only has to add.
  [`GH/models/vae_tiled.py#L58-L118`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/vae_tiled.py#L58-L118)
- Quality vs an exact fp32 untiled CPU decode: 44.67 dB. Overlap 3 or 2 was faster but lost
  1-2.5 dB, so it was not taken.

**Animation:** the 704x544 frame splits into 8 overlapping tiles, one per core, then the
overlaps blend back with linear ramps.

---

## Slide 8. VAE: kernels and fusion (6.0 to 2.6 s)

| Change | VAE |
|---|---|
| tiled, eager | 6.0 s |
| nkilib conv3d for the causal 3x3x3 convs | 5.4 s |
| upsample convs on nkilib | 5.2 s |
| rank-0 CPU threads 16, in-place uint8 convert | 4.2 s |
| RMS_norm + SiLU as one compiled graph | 3.5 s |
| streamed blend + mp4 encode during the decode | 3.0 s |
| head conv on nkilib + sub-pixel upsample + fused cache concat + residual in the kernel | 2.7 s |
| fused upsample shortcut, vfork the encoder | 2.6 s |

Three ideas worth a slide each:

1. **Small, targeted compiles inside an eager model.** The norm + SiLU pair is memory bound.
   One tiny compiled graph does it in one pass over memory instead of several.
   [`GH/wan/modules/vae2_2.py#L226-L228`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/wan/modules/vae2_2.py#L226-L228)
   ```python
   def _rms_silu_impl(x, gamma):
       rms = x.float().pow(2).mean(1, keepdim=True).clamp(min=1e-8).rsqrt()
       return F.silu((x * rms).to(gamma.dtype) * gamma)
   _rms_silu = torch.compile(_rms_silu_impl, backend="neuron", dynamic=False)
   ```
2. **Sub-pixel upsampling.** A nearest-2x upsample followed by a 3x3 conv is the same as four
   2x2 convs on the low-res input, one per output phase. That is 2.25x fewer MACs, and the 4x
   upsampled tensor is never built.
   [`GH/kernels/vae_conv3d.py#L60-L117`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L60-L117)
3. **Residual inside the kernel.** Each ResidualBlock's shortcut is added by the conv kernel
   itself (`residual_add_loc=POST_ACT`), which saves one HBM round trip and one launch.
   [`GH/kernels/vae_conv3d.py#L44-L49`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L44-L49)

Quality: 44.60 dB with the VAE flags on and 44.59 dB with them off, both against the exact
fp32 decode.

**Animation:** the sub-pixel trick. A 2x upsampled grid folds into four coloured 2x2 kernels.

---

## Slide 9. Host overhead counts too

The device is only fast if the host keeps it fed.

| Fix | What it removed |
|---|---|
| Rank 0 gets 16 CPU threads (torchrun sets 1) | slow CPU tile sum and uint8 convert. VAE 5.2 to 4.2 s |
| Stream the blend + mp4 encode per latent frame (`VAE_STREAM`) | 0.53 s that ran after the decode, now a 0.07-0.12 s tail |
| UniPC scheduler step on the device (`DEVICE_SCHEDULER`) | ~19 ms of device idle per step (D2H, CPU step, H2D, and waiting for the slowest rank). Device busy 96.9% to 99.5% |
| Spawn ffmpeg with vfork (`MP4_VFORK`) | one 88-95 ms stall per decode: `fork()` of a tens-of-GB process blocked the runtime's completion thread |

[`GH/models/device_unipc.py`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/device_unipc.py):
the scalar coefficients stay on the CPU and only the tensor combinations run on the device. It
is bit-identical to the CPU scheduler on the CPU, and the mp4 is byte-identical.

[`GH/inference_neuron_ti2v.py#L84-L88`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L84-L88)
explains the vfork fix.

**Talking point:** each of these was found in the system trace as a gap, not by reading code.

---

## Slide 10. Measure honestly

A true story from the week of the talk:

1. The validation job reported loop 31.4 s against main 29.5 s. That looked like a regression.
2. The cause was not the code. Run 2 was not warm: on this image, the NKI kernel cache still
   traces kernels during run 2. The VAE went 186 s, then 21 s, then 3.6 s.
3. Timing run 3 instead showed main at 11.8 s and loop at 7.1 s.

Other traps we hit:
- A floating `:latest` image changes the runtime under you. Record the image digest and SDK
  versions with every number.
- A container CPU limit throttles the host-side work: the VAE assembly, the launches, the mp4.
- A probe that replaced a reduce-scatter with a row slice let the compiler skip 3/4 of the
  matmul rows. It suggested collectives cost 6.2 ms when they actually cost 3.1 ms.

**Rule:** at least 3 runs, time the last ones, print every run, and pin the image.

**Animation:** three bars per config (run 1, 2, 3). The audience sees run 2 still "hot".

---

## Slide 11. Validate quality, not just speed

The validation job (`scripts/validate/validate_job.sh`) runs in one pod:

| Check | Result |
|---|---|
| New code with every flag off vs main: mp4 bytes | IDENTICAL |
| The same config twice: latent + frames | IDENTICAL |
| Step-0 noise prediction vs an fp32 CPU DiT | 0.0482 rel-RMS (bf16 floor ~0.048) |
| Whole video vs exact-SDPA run | 20.45 dB (main 20.68 dB) |
| VAE vs exact fp32 CPU decode | 44.60 dB (flags off 44.59 dB) |
| Contact sheet, frames 0/20/50/80 | checked by eye |

**Talking point:** bf16 sets an error floor. Each optimization must add nothing beyond it, and
the fp32 CPU references let you show that.

---

## Slide 12. Profiling workflow

1. **System trace on every rank** (`NEURON_RT_INSPECT_SYSTEM_PROFILE=1`). It is cheap, and it
   shows host gaps, launches and stalls (the 19 ms per step, the 90 ms fork).
2. **Device trace on one rank** (session mode). SPMD means one rank's device trace stands for all.
3. **Single-NEFF captures** (`capture --single-io`) for per-graph engine utilization.
4. **Neuron Explorer** with source roots, so an instruction links back to the torch line and
   the NKI kernel line. Use `kubectl port-forward <pod> 3001:3001 3002:3002`.

Scripts: [`scripts/prof/bench_profile.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/bench_profile.sh),
[`scripts/prof/rank_env.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/rank_env.sh),
[`scripts/prof/explorer_ingest.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/explorer_ingest.sh).

Where the 8.8 s goes today (RESULTS.md, profiled):

| Area | Time | Running at |
|---|---|---|
| DiT linear layers | 2.47 s | 83% of bf16 peak |
| DiT attention | 1.77 s | 45% of peak (self) |
| VAE convolutions | 1.74 s | 79% on the main convs |
| DiT collectives | 0.94-1.2 s | mostly serialized |
| DiT pointwise | 0.76 s | memory bound |
| VAE norms, small graphs | 0.56 s | memory bound |
| Host / idle | 0.57 s | |

Headroom estimate: about 2-3 s more without changing the output.

_To add: Explorer screenshots from the tp8 profiling job (device and system), with the hotspots
called out._

---

## Slide 13. Takeaways

1. Compile scope is a design choice. Go big for regular graphs (the DiT), and use small kernels
   for irregular ones (the VAE).
2. Use approximations only where you can prove they are safe (const-max per layer), and prefer
   an exact kernel that is just as fast (`attention_cte`).
3. Parallelism layout matters as much as kernels: CFG-parallel, SP, and local cross-attention.
4. The host is part of the model. Look for gaps in the system trace.
5. Measure honestly and validate quality to the bit.

---

## Open items before the talk

- Confirm which node the 7.1 s validation ran on (trn2 or trn3-dev1).
- Add Explorer screenshots from the tp8 profiling job (`788674b`).
- Decide slide count and time per slide (code talk, ~60 min).
- Then build the HTML deck with the animations described above.
