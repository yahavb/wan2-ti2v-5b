# Serving world models on AWS Trainium with native PyTorch

re:Invent code talk: outline, talking points and code to show.

> The next major leap in AI is predicting and interacting with the physical world. World models
> bridge digital reasoning with real-world execution, introducing new infrastructure demands for
> video generation, robotics, and gaming workloads. We'll explore the emerging architecture of
> world models and why AWS Trainium, purpose-built for better price-performance, is well suited
> to serve them. Learn how native PyTorch support with torch.compile fuses operations into
> optimized device graphs, and how NKI custom kernels accelerate critical operators like
> attention. You'll leave able to move world model inference from eager prototyping to compiled
> deployment using familiar PyTorch workflows.

**Worked example:** Wan2.2-TI2V-5B, a text + image to video model. One 704x544, 81-frame video,
10 denoising steps, on one Trainium2 device (8 NeuronCores). Numbers are from
[RESULTS.md](../../RESULTS.md). Code links are pinned to commit `788674b`.

**The arc:** eager prototype (76.6 s) → torch.compile (36.8 s) → NKI kernels → scale across
cores → serve it (8.7 s). Each stage uses PyTorch the way the audience already knows it.

---

## 1. World models and what they ask of the hardware (~8 min)

### Slide: what a world model is
- A model that predicts how a scene evolves: the next frames, given what it has seen and an
  instruction or action.
- Today the most visible form is video generation. Robotics and games use the same idea with
  actions as the condition.

### Slide: the architecture, using Wan2.2 as the concrete case
```
 text ──► T5 encoder ──────────────┐
                                   ▼
 image ─► VAE encoder ─► latent ─► Diffusion Transformer (DiT) × 10 steps ─► VAE decoder ─► video
                                   ▲  30 blocks, 7,854 video tokens
                                   └─ CFG: conditional + unconditional pass every step
```
- **Text encoder:** runs once per request.
- **DiT:** the core. Full attention over every video token (space and time). It runs 30 blocks
  x 10 steps x 2 CFG passes = 600 block calls per video.
- **VAE decoder:** a causal 3D conv net. It turns 21 latent frames into 81 frames of video.

### Slide: what this means for serving
1. **Long sequences.** Attention over thousands of tokens dominates compute.
2. **Many repeated calls.** The same block graph runs hundreds of times, so per-call overhead
   matters.
3. **Two very different halves.** The DiT is a regular, matmul-heavy transformer. The VAE is
   a long chain of convolutions whose shapes change from chunk to chunk.
4. **Latency per request.** Users wait for a whole video, so the time to the last frame is
   what counts.

Talking point: these four demands drive every decision in the rest of the talk.

---

## 2. Why Trainium for this workload (~5 min)

- One Trainium2 device exposes 8 NeuronCores in LNC1 mode. Each core has Tensor, Vector,
  Scalar and GpSimd engines, and the cores share fast in-device collectives.
- Every number in this talk uses one device and fits in its HBM: the 5B DiT, the T5 encoder
  and the VAE together.
- **Native PyTorch (TorchNeuron):** `neuron` is a regular torch device. Eager mode works, and
  `torch.compile(backend="neuron")` compiles to device graphs (NEFFs). NKI kernels are
  callable from PyTorch.
- Kubernetes on EKS: the pod requests NeuronCores with a resource claim, the same way it would
  request any other accelerator.

_Price-performance: quote only the official AWS figures here. This talk doesn't measure
cost._

---

## 3. Step 1: the eager prototype (~7 min)

**Message:** start in eager mode. It runs the reference model with very few changes, and it
gives you a correct baseline.

Live code: move the model to the device.
[`inference_neuron_ti2v.py#L72`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L72),
[`#L194-L213`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/inference_neuron_ti2v.py#L194-L213)

```python
NEURON_DEVICE = torch.device("neuron")
torch.neuron.set_device(local_rank)
text_encoder.model = text_encoder.model.to(NEURON_DEVICE)
vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
```

What we had to change in the reference code (keep this short; it is the honest list):
- No complex dtypes on the device. RoPE is rewritten with real cos and sin tables
  ([`rope_neuron.py`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/rope_neuron.py)).
- No float64 on the device.

Baseline: **76.6 s** per video (DiT 44.3 s, VAE 32.3 s). The output is correct, and it is the
reference that every later step is checked against.

Talking point: get it correct in eager mode first, and save that output. Every later
optimization must reproduce it.

---

## 4. Step 2: torch.compile fuses the model into device graphs (~12 min)

**Message:** torch.compile is where most of the speed comes from. What you put inside one
compiled graph matters as much as compiling at all.

### Slide: the first attempt, compiling pieces
The prototype compiled only small modules. The all-reduce after each tensor-parallel matmul was
kept outside the graph on purpose, so each block became many small graphs with Python in between.
[`models/tp_utils.py#L119-L128`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/tp_utils.py#L119-L128)

```python
block.ffn = torch.compile(block.ffn, backend="neuron", dynamic=False)

@torch.compiler.disable          # forces a graph break around every collective
def all_reduce_sum(x): ...
```

### Slide: one graph per transformer block
[`models/fast_dit.py#L260-L311`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L260-L311),
[`#L586-L645`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L586-L645)

```python
import torch.distributed._functional_collectives as funcol

def _all_reduce(x):
    return funcol.all_reduce(x, "sum", get_tp_group())   # traced: lives inside the graph

class _FastBlock(nn.Module):
    def forward(self, x, e, cos, sin, ck, cv):
        # norm + modulation, QKV, QK-norm, RoPE, attention, O-proj + all-reduce,
        # cross-attention, FFN + all-reduce, gated residuals: all in one graph
        ...

kw = dict(backend="neuron", dynamic=False, fullgraph=True)
self.blocks = [torch.compile(_FastBlock(b, ...), **kw) for b in model.blocks]
```

What the compiler does with that:
1. **Fusion.** Norms, modulation, RoPE and residual adds are fused with their neighbours,
   instead of a round trip to HBM for each op.
2. **One launch per block** instead of many, with no Python in between.
3. **Collectives scheduled by the compiler**, next to the matmul that produces them.
4. **One NEFF for all 30 blocks.** The weights are graph inputs, so one compile serves all
   600 block calls.
5. **`fullgraph=True` fails loudly** if anything would split the block.

Also: batch the two CFG passes into one call.

Result: DiT 4.4 → 1.91 s/step. Total **76.6 s → 36.8 s**.

**Demo idea:** the same block, eager vs compiled, in the Neuron Explorer system trace. Many
short executions with gaps become one long execution.

### Slide: compile is not always the answer (the VAE)
| Attempt | VAE time | Compile time |
|---|---|---|
| torch.compile across the whole VAE | 400+ graphs, out of memory | — |
| torch.compile of the decoder | 11.5-12.9 s | 58 min |
| Static-shape decoder, 2 graphs | 12.8-14.2 s | 20.6 min |
| Eager + NKI kernels | **1.7 s** | none |

(At 368x480x21.) The causal cache changes shapes from chunk to chunk, and the compiler's 3D
convolutions were slow. That sets up the next section.

Use small compiled graphs where they help: the VAE's RMS_norm + SiLU is one tiny compiled
function inside the eager model.
[`wan/modules/vae2_2.py#L226-L228`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/wan/modules/vae2_2.py#L226-L228)

```python
_rms_silu = torch.compile(_rms_silu_impl, backend="neuron", dynamic=False)
```

---

## 5. Step 3: NKI custom kernels for the critical operators (~12 min)

**Message:** when one operator dominates and the compiler's version is not good enough, write
or reuse an NKI kernel and call it from PyTorch. This works in eager mode and inside a
compiled graph.

### Slide: what NKI is, in one slide
- Neuron Kernel Interface: Python kernels that control the tiling, the on-chip memory (SBUF)
  and which engine does what.
- nkilib ships production kernels, such as `attention_cte` and `conv3d`, that you can use
  directly.
- `wrap_nki` makes a kernel a regular torch call.

### Slide: attention, the operator that dominates
At 7,854 tokens, self-attention through the default SDPA path took 18.2 ms per call.

**Kernel 1, const-max softmax (5.8 ms per call).** It subtracts a known bound instead of
tracking a running max. That is only exact when scores stay under the bound, and on all layers
the video broke after frame 0. The fix was to compute each layer's worst-case score from its
QK-norm weights and use the kernel only where it is safe (22 of 30 layers).
[`models/fast_dit.py#L77-L84`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L77-L84)

```python
def const_max_layer_ok(block):
    gq = sa.norm_q.weight.abs().reshape(-1, d).amax(1)
    gk = sa.norm_k.weight.abs().reshape(-1, d).amax(1)
    hard = float((d ** 0.5) * (gq * gk).max())   # worst-case scaled score
    return hard <= CONST_MAX_HARD_LIMIT, hard
```
DiT 1.92 → 1.19 s/step.

**Kernel 2, nkilib `attention_cte` (5.3 ms per call).** It is exact flash attention with no
bound, so it is safe on every layer. It runs inside the compiled block graph.
[`models/fast_dit.py#L228-L241`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L228-L241)

```python
from nkilib.core.attention.attention_cte import attention_cte

def _cte(q, k, v):
    qn = q.permute(0, 2, 1, 3).reshape(B * n, Lq, d)
    kn = k.permute(0, 2, 3, 1).reshape(B * n, d, Lk)
    vn = v.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    return attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False)
```
Self- and cross-attention: total 16.2 → 13.2 s.

Talking points:
- Validate a numerical shortcut on real activations, not on random inputs.
- Kernel limits are real. Calling const-max on all heads at once overflowed SBUF and crashed
  the compiler, so it is called on as many heads as fit.

### Slide: the same approach for the VAE convolutions
nkilib `conv3d`, called from the eager VAE, with a gate that decides when it beats the
default.
[`kernels/vae_conv3d.py#L33-L58`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/kernels/vae_conv3d.py#L33-L58)

```python
@nki.jit
def _conv3d_3x3x3_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1))

_conv3d_3x3x3 = wrap_nki(_conv3d_3x3x3_kernel)
```
Then fuse into the kernel: the residual add (`residual_add_loc=POST_ACT`), and an upsample
followed by a conv done as four 2x2 convs on the low-res input (2.25x fewer MACs). VAE 6.0 →
2.6 s.

---

## 6. Step 4: scale across the cores with torch.distributed (~6 min)

**Message:** the parallelism is plain `torch.distributed` and functional collectives, traced
into the compiled graphs.

| Layout | Per DiT block | Total |
|---|---|---|
| Tensor parallel over 8 cores, CFG as a batch of 2 | 29.4 ms | 13.2 s |
| CFG-parallel: two groups of 4, one per CFG pass | 24.5 ms | 11.9 s |
| + sequence parallelism (all-gather / reduce-scatter around the matmuls) | 21.9 ms | 11.1 s |
| + cross-attention local to each core's tokens | 20.6 ms | 10.8 s |

[`models/fast_dit.py#L314-L390`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/fast_dit.py#L314-L390)

```python
hf = _ag(h)                          # funcol.all_gather_tensor: token shard -> all tokens
q, k, v = _lin(hf, sa.q), _lin(hf, sa.k), _lin(hf, sa.v)
...
y = _rs(_mm(y, sa.o)) + sa.o.bias    # funcol.reduce_scatter_tensor: back to the shard
```

The VAE is split the same way: each core decodes one spatial tile and rank 0 blends them (VAE
9.0 → 6.0 s).
[`models/vae_tiled.py#L58-L118`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/vae_tiled.py#L58-L118)

---

## 7. Step 5: from benchmark to serving (~8 min)

**Message:** the compiled model is only useful if the server starts warm, keeps the device
fed, and is deployed like any other pod.

### Slide: the server
[`serve_ti2v.py`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/serve_ti2v.py):
FastAPI on rank 0 (`POST /generate`, `/health`, `/readiness`), with all 8 ranks in one
torchrun.
- Collectives must run on the thread that compiled them. HTTP requests go on a queue, and the
  main thread runs them.
- Requests use the warm-up shape, so no request triggers a compile.
- `/readiness` turns true only after warm-up, so Kubernetes sends no traffic to a cold pod.

### Slide: warm-up is part of deployment
A lesson from this week: one warm-up run was not enough. On the current image, the second run
still traced NKI kernels. The VAE took 186 s, then 21 s, then 3.6 s. Warm up until the
timing settles before reporting ready.

### Slide: keep the device fed
Fixes found as gaps in the system trace:
- UniPC scheduler step on the device instead of a round trip to the CPU each step (device busy
  96.9% → 99.5%)
  ([`device_unipc.py`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/models/device_unipc.py)).
- Encode the mp4 while the VAE is still decoding, instead of after it.
- Start ffmpeg with vfork: forking a process with tens of GB of memory stalled the runtime for
  90 ms.

### Slide: deploy on EKS
[`wan2-ti2v-5b-deploy.yaml`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/wan2-ti2v-5b-deploy.yaml):
a resource claim for the NeuronCores, model weights cached on a PVC, and readiness and
liveness probes.

---

## 8. Did it stay correct, and where did the time go (~5 min)

**Validate every step against the eager reference**
([`scripts/validate/validate_job.sh`](https://github.com/yahavb/wan2-ti2v-5b/blob/788674b/scripts/validate/validate_job.sh)):
- Byte-identical output with the new flags off, and identical output across repeated runs.
- Step-0 noise prediction against an fp32 CPU run: within the bf16 floor.
- VAE against an exact fp32 decode: 44.6 dB.

**Profile with Neuron Explorer:** a system trace on every rank shows host gaps, and a device
trace on one rank shows the engines, linked back to the source.

The journey:

| Stage | Total |
|---|---|
| Eager prototype | 76.6 s |
| + torch.compile, one graph per block | 36.8 s |
| + VAE tiled over 8 cores, NKI conv kernels | 16.1 s |
| + NKI attention, CFG-parallel, sequence parallel | 10.8 s |
| + VAE fusion, host fixes | **8.7 s** |

For reference, an A100 with torch.compile takes 16.92 s for the same video.

---

## 9. Takeaways (~2 min)

1. Start in eager mode on the `neuron` device. It gives you a correct baseline with minimal
   changes.
2. Compile with `torch.compile(backend="neuron")`, and choose the scope: one graph per repeated
   block, with the collectives inside it.
3. Where one operator dominates, use an NKI kernel from PyTorch. Check its numerics on real
   data.
4. Scale with plain `torch.distributed` and functional collectives.
5. Serve warm: warm up until the timing settles, gate readiness on it, and keep the host out of
   the way.

---

## Open items

- **Serving gap:** `serve_ti2v.py` still uses the old per-module compile path (lines 137-141),
  not `fast_dit`, and it warms up only once. Port it before the talk, or present serving as the
  next step.
- Confirm which node the 7.1 s validation run used (trn2 or trn3-dev1). This outline uses only
  trn2 numbers.
- Add Explorer screenshots: eager vs compiled block, and attention SDPA vs `attention_cte`.
- Get the official Trainium price-performance wording.
