# A skeleton for inference optimization, worked end to end on Wan2.2-TI2V-5B

re:Invent code talk: talking points and slide plan. Every number comes from
[RESULTS.md](../../RESULTS.md). Code links are pinned to commit `788674b`.

Spec for every run: 704x544, 81 frames, 10 denoising steps, CFG on, the same prompt, image
and seed, one trn2 NeuronDevice exposed as 8 logical cores (LNC1). Times are warm runs.

**The story in one line:** a repeatable loop (measure, validate, profile, pick a lever,
change one thing, record) took one video from 76.6 s to 8.7 s on the same hardware. The talk
teaches the loop. The model is the worked example.

---

## Part 1. The skeleton

### Slide 1. Where we started and where we ended

- Start: the reference model ported as is, eager PyTorch, 76.6 s per video.
- End: 8.7 s, 8.8x faster, same hardware, same output quality.
- Baseline for context: the same video on an A100 with torch.compile takes 16.92 s. This is the
  only time the talk mentions it.

**Talking point:** none of the steps were exotic. What made it work was the order and the
discipline.

**Animation:** one bar shrinking from 76.6 s to 8.7 s, with each milestone appearing as a tick.

### Slide 2. The loop

```
        ┌──────────────────────────────────────────────┐
        ▼                                              │
  0. trustworthy measurement + quality gates          │
  1. profile: where does the time go?                 │
  2. pick the lever for the biggest bucket            │
  3. change one thing behind a flag                   │
  4. validate: still the same output?                 │
  5. record it (commit + one row in RESULTS.md) ──────┘
```

**Talking point:** set steps 0 and 4 up before the first optimization, not after. Every
later decision depends on trusting the numbers and the output.

**Animation:** the loop drawn once, then highlighted each time a later slide goes around it.

### Slide 3. Step 0: make the numbers trustworthy

What the harness does (`inference_neuron_ti2v.py`, `scripts/prof/bench_profile.sh`):
1. Same spec, seed and inputs for every run.
2. Several runs in one process. Time only the runs after warm-up, and print every run.
3. Split the time into phases (denoise, VAE, output) so you know which part moved.
4. Record the commit, the image and the flags next to every number.

A story from the week of this talk:
- A validation job showed the new code at 31.4 s against 29.5 s for the old one. That looked
  like a regression.
- But run 2 was not warm. On this image, the NKI kernel cache still traces kernels during
  run 2. The VAE went 186 s, then 21 s, then 3.6 s.
- Timing run 3 instead showed 11.8 s and 7.1 s, the opposite conclusion.

Other traps:
- A floating `:latest` image changes the runtime under you.
- A container CPU limit throttles host-side work.
- A probe that removes "just the collective" can let the compiler skip real work too.

**Animation:** three bars per config (runs 1, 2, 3). Run 2 is still hot.

### Slide 4. Step 0: quality gates before any speedup

Build the references once and reuse them for every change
(`scripts/validate/validate_job.sh`):

| Gate | Why |
|---|---|
| Every new flag off: mp4 bytes identical to main | the refactor alone changed nothing |
| Same config twice: identical latent and frames | the run is deterministic, so diffs mean something |
| Step-0 noise prediction vs an fp32 CPU DiT | each change adds nothing beyond the bf16 floor (~0.048 rel-RMS) |
| VAE vs an exact fp32 CPU decode | tiling and fusion error in dB (44.6 dB) |
| Whole video vs an exact-attention run, plus a contact sheet | catches drift a single step can't show |

**Talking point:** every optimization ships behind an env flag, defaulting to off. Then "flag
off equals main" is a byte compare, and A/B testing is one env var.

### Slide 5. Step 1: map the model before you profile it

- Three models in a pipeline: T5 text encoder, a 30-block DiT (7,854 tokens, dim 3072), and a
  causal 3D VAE decoder.
- Count the work: 30 blocks x 10 steps x 2 CFG passes = 600 block calls per video. The VAE
  decodes 21 latent frames into 81 video frames.
- The first split already tells you where to look: DiT 44.3 s and VAE 32.3 s out of 76.6 s.

**Talking point:** different components need different levers. The DiT is a large, regular,
repeated graph. The VAE is a long chain of convolutions whose shapes change from chunk to chunk.

### Slide 6. Step 1: profile, top down

1. **System trace on every rank** (cheap). It shows device busy vs idle, launches, host gaps
   and stalls. Start here.
2. **Device trace on one rank.** SPMD means one rank stands for all of them. It shows engine
   time inside each graph.
3. **Single-graph captures** (`capture --single-io`) for per-graph engine utilization.
4. **Neuron Explorer** with source roots, so an instruction links back to the torch line or
   the NKI kernel line.

Scripts: [`bench_profile.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/bench_profile.sh),
[`rank_env.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/rank_env.sh),
[`explorer_ingest.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/prof/explorer_ingest.sh).

_To add: Explorer screenshots, system and device._

### Slide 7. Step 2: from symptom to lever

| What the trace shows | Lever |
|---|---|
| Many short graphs, gaps between them | **Compile scope**: put more into one graph |
| Compile explodes (many graphs, OOM) or compiled code is slow | **Hand-written kernels** in an eager model |
| One op dominates (attention, conv) at low engine utilization | **A better kernel** for that op |
| Cores idle, or collectives everywhere | **Parallel layout**: what each core does |
| Memory-bound pointwise ops between big ops | **Fusion** |
| Device idle while the host works | **Host work**: threads, async, move work to the device |

**Talking point:** this table is the skeleton. The rest of the talk walks through it once for
each lever, in the order the profile asked for them.

---

## Part 2. Going around the loop

### Slide 8. Lever: compile scope (DiT, 76.6 s to 36.8 s)

**Symptom:** each block ran as many small graphs. Every TP all-reduce forced a graph break on
purpose, and Python ran in between.

[`models/tp_utils.py#L119-L128`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/tp_utils.py#L119-L128)

```python
@torch.compiler.disable
def all_reduce_sum(x):  # graph break: compiled NEFF does the matmuls, all-reduce runs eager
```

**Change (`DIT_FAST=1`):** one graph per transformer block, collectives traced inside it, and
both CFG passes in one batch.

[`models/fast_dit.py#L90-L100`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L90-L100),
[`#L260-L311`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L260-L311)

```python
def _all_reduce(x):
    return funcol.all_reduce(x, "sum", get_tp_group())      # stays inside the NEFF

kw = dict(backend="neuron", dynamic=False, fullgraph=FAST_FULLGRAPH)
self.blocks = [torch.compile(_FastBlock(b, ...), **kw) for b in model.blocks]
```

**Why it helps:** one launch per block, fusion across neighbouring ops, collectives scheduled
next to their matmuls, and one NEFF shared by all 30 blocks (the weights are inputs).

**Result:** 4.4 to 1.91 s/step.

**Animation:** one block as a timeline. Many short boxes with gaps collapse into one long box.

### Slide 9. Lever: compile scope fails; switch to kernels (VAE)

**Symptom:** the VAE was now the biggest bucket. The same lever made it worse.

| Attempt | VAE | Compile |
|---|---|---|
| `torch.compile` across the whole VAE | 400+ NEFFs, out of memory | — |
| `torch.compile` of the original decoder | 11.5-12.9 s | 58 min |
| Static-cache decoder, 2 graphs | 12.8-14.2 s | 20.6 min |
| **Eager + NKI kernels, tiled** | **1.7 s** | none |

(Measured at 368x480x21, where the VAE dominated.)

**Change:** keep the VAE eager, and call hand-written kernels from it: conv2d as shifted
matmuls, VAE attention, then nkilib `conv3d` for the causal 3x3x3 convs.

[`kernels/vae_conv3d.py#L33-L58`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L33-L58)

```python
@nki.jit
def _conv3d_3x3x3_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1))

_conv3d_3x3x3 = wrap_nki(_conv3d_3x3x3_kernel)   # callable from eager PyTorch
```

**Lesson:** a kernel needs a gate. This one is used only when C_in and C_out >= 96 and there
are at least 2 output frames. Below that, the eager conv was faster, and 1-frame convs regressed
the decode by 25%.

**Animation:** a compile-time bar (58 min) next to a latency bar (12.9 s vs 1.7 s).

### Slide 10. Lever: parallel layout (VAE tiles, 9.0 to 6.0 s)

**Symptom:** the VAE ran on rank 0 while 7 cores sat idle.

**Change:** every rank decodes an overlapping spatial tile. It applies a blend weight that is
already normalized, on the device, then writes to `/dev/shm`. Rank 0 only adds.
[`models/vae_tiled.py#L58-L118`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/vae_tiled.py#L58-L118)

**Trade-off, measured:** overlap 4 gives 44.67 dB against the exact decode. Overlaps 3 and 2
were faster but cost 1-2.5 dB, so they were not taken.

**Animation:** the frame splits into 8 tiles, one per core, and blends back.

### Slide 11. Lever: a better kernel, used where it is safe (attention, 1.92 to 1.19 s/step)

**Symptom:** self-attention dominated the DiT. SDPA took 18.2 ms per call.

**Change:** a const-max softmax kernel that subtracts a known bound instead of tracking a
running max (5.8 ms per call).

**What went wrong:** on all layers, the video broke after frame 0.

**Fix:** bound each layer's worst-case score from its QK-norm weights, and use the kernel only
where the bound is <= 70. That is 22 of 30 layers.
[`models/fast_dit.py#L77-L84`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L77-L84)

```python
def const_max_layer_ok(block):
    gq = sa.norm_q.weight.abs().reshape(-1, d).amax(1)
    gk = sa.norm_k.weight.abs().reshape(-1, d).amax(1)
    hard = float((d ** 0.5) * (gq * gk).max())   # worst-case scaled score
    return hard <= CONST_MAX_HARD_LIMIT, hard
```

**Two more lessons:**
- Calling the kernel on all heads at once overflowed SBUF and crashed the compiler. The fix
  was to call it on as many heads as fit
  ([`#L210-L225`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L210-L225)).
- Take the layer choice as a max over ranks. If ranks pick different layers, they compile
  different graphs and their collectives deadlock.

**Then go around the loop again:** nkilib `attention_cte` is exact, needs no bound, and runs
at 5.3 ms. It replaced const-max on every layer (16.2 to 13.2 s with cross-attention too).
[`#L228-L241`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L228-L241)

### Slide 12. Lever: parallel layout (DiT, 13.2 to 9.6 s)

**Symptom:** with attention fixed, the collectives and the work repeated on every core showed
up in the profile.

| Change | Per block | Total |
|---|---|---|
| TP=8, CFG as a batch of 2 | 29.4 ms | 13.2 s |
| CFG-parallel: 2 groups x TP4, batch 1 each | 24.5 ms | 11.9 s |
| + Megatron sequence parallelism | 21.9 ms | 11.1 s |
| + cross-attention local to the token shard | 20.6 ms | 10.8 s |
| + CFG groups on alternating cores | 19.8 ms | 9.6 s |

- **CFG-parallel:** the conditional and unconditional passes run at the same time on half the
  cores each. Smaller TP means cheaper collectives.
  [`inference_neuron_ti2v.py#L241-L246`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L241-L246)
- **Sequence parallelism:** an all-gather before each column-parallel matmul and a
  reduce-scatter after each row-parallel matmul. That moves the same bytes as the all-reduce,
  and the norms and residuals run on 1/4 of the tokens.
  [`models/fast_dit.py#L314-L390`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L314-L390)
  ```python
  hf = _ag(h)                          # token shard -> all tokens
  q, k, v = _lin(hf, sa.q), _lin(hf, sa.k), _lin(hf, sa.v)
  y = _rs(_mm(y, sa.o)) + sa.o.bias    # all tokens -> token shard
  ```
- **Local cross-attention:** there are only 512 text keys, so each rank runs all heads on its own
  tokens, with no collectives.

(The VAE fused-norm step, slide 13, lands between rows 4 and 5.)

**Animation:** 8 cores. First one batch, then two colours (cond / uncond), then token shards.

### Slide 13. Lever: fusion (VAE, 4.2 to 2.6 s)

**Symptom:** between the big convs, the trace showed many small memory-bound graphs.

1. **A small compiled graph inside an eager model.** RMS_norm + SiLU becomes one pass over memory.
   [`wan/modules/vae2_2.py#L226-L228`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/wan/modules/vae2_2.py#L226-L228)
   ```python
   _rms_silu = torch.compile(_rms_silu_impl, backend="neuron", dynamic=False)
   ```
2. **Change the math, not just the code.** A nearest-2x upsample followed by a 3x3 conv equals
   four 2x2 convs on the low-res input. That is 2.25x fewer MACs, and the upsampled tensor is
   never built.
   [`kernels/vae_conv3d.py#L60-L117`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L60-L117)
3. **Fold neighbours into the kernel.** The ResidualBlock shortcut is added inside the conv
   (`residual_add_loc=POST_ACT`), and the causal-cache concat is fused into the norm graph.
   [`kernels/vae_conv3d.py#L44-L49`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L44-L49)

**Validation:** 44.60 dB with the fusions on and 44.59 dB with them off, against the exact
fp32 decode.

**Animation:** the sub-pixel fold. A 2x grid becomes four coloured 2x2 kernels.

### Slide 14. Lever: host work

**Symptom:** the device sat idle while the host worked. Each of these was found as a gap in
the system trace.

| Gap in the trace | Fix |
|---|---|
| Slow CPU tile sum and convert on rank 0 (torchrun sets 1 thread) | 16 threads on rank 0. VAE 5.2 to 4.2 s |
| 0.53 s after the decode for blend + mp4 | stream both per latent frame during the decode |
| ~19 ms per step: D2H, CPU scheduler step, H2D, waiting for the slowest rank | UniPC step on the device ([`device_unipc.py`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/device_unipc.py)). Busy 96.9% to 99.5% |
| One 90 ms stall per decode | `fork()` of a tens-of-GB process to start ffmpeg; spawn it with vfork ([`#L84-L88`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L84-L88)) |

**Talking point:** the host is part of the model. These fixes are small, but they matter more
the faster the device gets.

---

## Part 3. Knowing when to stop

### Slide 15. Where the time goes now, and what is left

From the profile of the current build (RESULTS.md):

| Area | Time | Running at |
|---|---|---|
| DiT linear layers | 2.47 s | 83% of bf16 peak |
| DiT attention | 1.77 s | 45% of peak (self) |
| VAE convolutions | 1.74 s | 79% on the main convs |
| DiT collectives | 0.94-1.2 s | mostly serialized with compute |
| DiT pointwise | 0.76 s | memory bound |
| VAE norms, small graphs | 0.56 s | memory bound |
| Host / idle | 0.57 s | |

**Talking point:** compare each bucket with its floor (the same work at 100% of peak). The
matmuls are near their floor, so stop there. Attention, collective overlap and fusion still
have about 2-3 s of headroom. The loop tells you where to go next, and also when a lever
is used up.

### Slide 16. Also record what didn't work

From RESULTS.md: chunked all-reduces, `-O3`, pre-transposed weights, fused QKV, runtime
collective knobs, 4 frames per VAE call (out of HBM), exact seam-free tiling (the halo exchange
cost as much as it saved).

**Talking point:** a negative result is still a row in the ledger. It stops the next person
from trying it again.

### Slide 17. The skeleton, to take home

1. Fix the spec, then build a harness that times only warm runs and records provenance.
2. Build quality gates first: byte-identical with flags off, deterministic, fp32 references.
3. Map the model: components, call counts, the first time split.
4. Profile top down: system trace first, then the device trace.
5. Map the symptom to a lever: compile scope, kernels, parallel layout, fusion, host.
6. Change one thing behind a flag, validate, and record a row. Then repeat.
7. Stop on a lever when it reaches its floor, and move to the next bucket.

---

## Open items before the talk

- Confirm which node the 7.1 s / 11.8 s validation ran on (trn2 or trn3-dev1). Use only
  trn2 numbers until then.
- Add Explorer screenshots from the tp8 profiling job (`788674b`) to slides 6, 8 and 14.
- Set slide timing for a ~60 min code talk.
- Then build the HTML deck with the animations.
