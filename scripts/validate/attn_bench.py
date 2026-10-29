"""Per-rank attention microbench at the Wan2.2-TI2V-5B 704x544x81 TP=8 shape (one NeuronCore).

Each variant runs inside its own torch.compile(backend="neuron") graph, like the DiT block.
q/k come from RMS-normalized random activations times a real layer's QK-norm gammas (so the
score range matches that layer); v is random. Error is vs an fp32 CPU reference.
Usage: python3 attn_bench.py [variant ...]   (default: sdpa constmax cte nst x_sdpa x_cte)
  self:  sdpa constmax cte nst | cte_tpk (tp_k=True) cte_sbf16 (bf16 softmax) cte_kern (kernel only,
         inputs pre-permuted on the host)
  cross: x_sdpa x_cte x_cmax (const-max kernel) x_mm32 / x_mmf (bmm + fp32 softmax) x_cte_tpk x_cte_kern
Env: LAYER (default 25), REPS (default 20), L (default 7854), LK_CROSS (512), LQ_CROSS (L),
     B x N = batch x heads per call (default 2 x 3; CFG-parallel TP4 self-attn 1 x 6, local cross 1 x 24
     with LQ_CROSS=1964)
"""
import glob
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
LAYER = int(os.environ.get("LAYER", "25"))
REPS = int(os.environ.get("REPS", "20"))
L = int(os.environ.get("L", "7854"))
LK_CROSS = int(os.environ.get("LK_CROSS", "512"))
# B x N = batch x heads per call. Default 2 x 3 (TP8 batch-2); the CFG-parallel TP4 self-attention
# is 1 x 6 (same kernel shapes), the local cross-attention 1 x 24 with LQ_CROSS=1964 (token shard).
B, N = int(os.environ.get("B", "2")), int(os.environ.get("N", "3"))
D, DIM, TP = 128, 3072, 8
LQ_CROSS = int(os.environ.get("LQ_CROSS", str(L)))
dev = torch.device("neuron")


def sync():
    torch.neuron.synchronize()


def gammas(layer, kind="self_attn"):
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "diffusion_pytorch_model*.safetensors"))):
        with safe_open(f, "pt") as st:
            for n in (f"blocks.{layer}.{kind}.norm_q.weight", f"blocks.{layer}.{kind}.norm_k.weight"):
                if n in st.keys():
                    out[n.split(".")[-2]] = st.get_tensor(n).float()
    return out["norm_q"], out["norm_k"]


def make_qkv(layer, lq, lk, kind="self_attn", seed=0):
    g = torch.Generator().manual_seed(seed)
    gq, gk = gammas(layer, kind)
    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    q = rms(torch.randn(B, lq, DIM, generator=g), gq)[..., :N * D].reshape(B, lq, N, D)
    k = rms(torch.randn(B, lk, DIM, generator=g), gk)[..., :N * D].reshape(B, lk, N, D)
    v = torch.randn(B, lk, N, D, generator=g)
    return q, k, v


def ref_attn(q, k, v):
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(N):
            s = (q[b, :, h] @ k[b, :, h].T) / math.sqrt(D)
            out[b, :, h] = s.softmax(-1) @ v[b, :, h]
    return out


# ── variants: q/k/v [B, L, n, d] bf16 -> [B, L, n, d] ──
def v_sdpa(q, k, v):
    from models.fast_dit import _sdpa
    return _sdpa(q, k, v)


def v_constmax(q, k, v):
    from models.fast_dit import _const_max_self
    return _const_max_self(q, k, v)


def v_cte(q, k, v):
    from nkilib.core.attention.attention_cte import attention_cte
    b, lq, n, d = q.shape
    lk = k.shape[1]
    qn = q.permute(0, 2, 1, 3).reshape(b * n, lq, d)
    kn = k.permute(0, 2, 3, 1).reshape(b * n, d, lk)
    vn = v.permute(0, 2, 1, 3).reshape(b * n, lk, d)
    out = attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False)
    return out.view(b, n, lq, d).transpose(1, 2)


def v_nst(q, k, v):
    from models.fast_dit import _nki_self
    return _nki_self(q, k, v)


def _cte_kw(tp_k=False, **kw):
    """attention_cte with extra kernel arguments (dtype / layout knobs)."""
    def f(q, k, v):
        from nkilib.core.attention.attention_cte import attention_cte
        b, lq, n, d = q.shape
        lk = k.shape[1]
        qn = q.permute(0, 2, 1, 3).reshape(b * n, lq, d)
        if tp_k:  # (bs, seqlen_kv, d): no transpose of k before the kernel
            kn = k.permute(0, 2, 1, 3).reshape(b * n, lk, d)
        else:
            kn = k.permute(0, 2, 3, 1).reshape(b * n, d, lk)
        vn = v.permute(0, 2, 1, 3).reshape(b * n, lk, d)
        out = attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False, tp_k=tp_k, **kw)
        return out.view(b, n, lq, d).transpose(1, 2)
    return f
def _mm_attn(fp32_inputs):
    """Plain batched matmuls + fp32 softmax: scores in fp32 (bmm out_dtype, or fp32 inputs)."""
    def f(q, k, v):
        b, lq, n, d = q.shape
        lk = k.shape[1]
        qh = q.permute(0, 2, 1, 3).reshape(b * n, lq, d)
        kh = k.permute(0, 2, 3, 1).reshape(b * n, d, lk)
        vh = v.permute(0, 2, 1, 3).reshape(b * n, lk, d)
        if fp32_inputs:
            s = torch.bmm(qh.float(), kh.float())
        else:
            s = torch.bmm(qh, kh, out_dtype=torch.float32)
        p = (s * d ** -0.5).softmax(-1).to(q.dtype)
        return torch.bmm(p, vh).view(b, n, lq, d).transpose(1, 2)
    return f
def _cmax_any(q, k, v):
    """Const-max kernel (kernels/attention_const_max.py: no running max, no P transpose) for any
    Lq, Lk (the DiT wrapper _const_max_self assumes Lq == Lk). Heads grouped per SBUF budget."""
    from kernels.attention_const_max import attention_const_max
    from models.fast_dit import CONST_MAX_SBUF_BUDGET, CONST_MAX_SOFTMAX_MAX
    b, lq, n, d = q.shape
    lk = k.shape[1]
    qn = q.permute(0, 2, 3, 1).reshape(b * n, d, lq)
    kn = k.permute(0, 2, 3, 1).reshape(b * n, d, lk)
    vn = v.permute(0, 2, 1, 3).reshape(b * n, lk, d)
    per_head = d * lq * 2 + 128 * (-(-lq // 128)) * (d + 1) * 4
    g = max(1, min(b * n, CONST_MAX_SBUF_BUDGET // per_head))
    outs = [attention_const_max(qn[i:i + g], kn[i:i + g], vn[i:i + g], CONST_MAX_SOFTMAX_MAX, d ** -0.5)
            for i in range(0, b * n, g)]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return out.view(b, n, lq, d).transpose(1, 2)
def _nl():
    import nki.language as nl
    return nl
VARIANTS = {"sdpa": v_sdpa, "constmax": v_constmax, "cte": v_cte, "nst": v_nst,
            "x_sdpa": v_sdpa, "x_cte": v_cte,
            "x_cmax": _cmax_any,
            "x_mm32": lambda q, k, v: _mm_attn(False)(q, k, v),
            "x_mmf": lambda q, k, v: _mm_attn(True)(q, k, v),
            "x_cte_tpk": lambda q, k, v: _cte_kw(tp_k=True)(q, k, v),
            "cte_tpk": lambda q, k, v: _cte_kw(tp_k=True)(q, k, v),
            "cte_sbf16": lambda q, k, v: _cte_kw(softmax_dtype=_nl().bfloat16)(q, k, v),
            "cte_mmbf16": lambda q, k, v: _cte_kw(mm_out_dtype=_nl().bfloat16)(q, k, v),
            "cte_both16": lambda q, k, v: _cte_kw(softmax_dtype=_nl().bfloat16,
                                                  mm_out_dtype=_nl().bfloat16)(q, k, v)}
# Kernel-only: q/k/v handed to the graph already in the kernel's layouts (permuted on the host),
# so the time excludes the permute copies the DiT block does around the kernel.
KERNEL_ONLY = {"cte_kern", "x_cte_kern"}
def bench_kernel_only(name, q, k, v, ref):
    from nkilib.core.attention.attention_cte import attention_cte
    b, lq, n, d = q.shape
    lk = k.shape[1]
    qn = q.permute(0, 2, 1, 3).reshape(b * n, lq, d).contiguous()
    kn = k.permute(0, 2, 3, 1).reshape(b * n, d, lk).contiguous()
    vn = v.permute(0, 2, 1, 3).reshape(b * n, lk, d).contiguous()
    ref_n = ref.permute(0, 2, 1, 3).reshape(b * n, lq, d).contiguous()
    def fn(qn, kn, vn):
        return attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False)
    bench(name, fn, qn, kn, vn, ref_n, flops_shape=(q.shape[1], k.shape[1]))


def bench(name, fn, q, k, v, ref, flops_shape=None):
    f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
    qd, kd, vd = (t.to(torch.bfloat16).to(dev) for t in (q, k, v))
    t0 = time.perf_counter()
    try:
        out = f(qd, kd, vd)
        sync()
    except Exception as e:
        print(f"{name:>9}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
        return
    t_first = time.perf_counter() - t0
    for _ in range(3):
        f(qd, kd, vd)
    sync()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(qd, kd, vd)
    sync()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    o = out.float().cpu()
    err = o - ref
    rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    lq_, lk_ = flops_shape or (q.shape[1], k.shape[1])
    flops = 4 * B * N * lq_ * lk_ * D
    print(f"{name:>9}: {ms:8.3f} ms/call  {flops / ms / 1e9:6.1f} TFLOP/s  rel_rms={rel:.5f} "
          f"max_abs={err.abs().max().item():.4f} finite={bool(torch.isfinite(o).all())} "
          f"(first call {t_first:.1f}s)", flush=True)


def main():
    names = sys.argv[1:] or ["sdpa", "constmax", "cte", "nst", "x_sdpa", "x_cte"]
    torch.set_grad_enabled(False)
    self_names = [n for n in names if not n.startswith("x_")]
    cross_names = [n for n in names if n.startswith("x_")]
    if self_names:
        q, k, v = make_qkv(LAYER, L, L)
        smax = max(((q[b, :, h] @ k[b, :, h].T) / math.sqrt(D)).max().item()
                   for b in range(B) for h in range(N))
        ref = ref_attn(q, k, v)
        print(f"self-attn layer {LAYER}: B*n={B * N} L={L} d={D} sample max scaled score {smax:.1f}", flush=True)
        for n in self_names:
            if n in KERNEL_ONLY:
                bench_kernel_only(n, q, k, v, ref)
            else:
                bench(n, VARIANTS[n], q, k, v, ref)
    if cross_names:
        q, k, v = make_qkv(LAYER, LQ_CROSS, LK_CROSS, kind="cross_attn")
        ref = ref_attn(q, k, v)
        print(f"cross-attn layer {LAYER}: B*n={B * N} Lq={LQ_CROSS} Lk={LK_CROSS} d={D}", flush=True)
        for n in cross_names:
            if n in KERNEL_ONLY:
                bench_kernel_only(n, q, k, v, ref)
            else:
                bench(n, VARIANTS[n], q, k, v, ref)


if __name__ == "__main__":
    main()
