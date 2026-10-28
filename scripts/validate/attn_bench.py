"""Per-rank attention microbench at the Wan2.2-TI2V-5B 704x544x81 TP=8 shape (one NeuronCore).

Each variant runs inside its own torch.compile(backend="neuron") graph, like the DiT block.
q/k come from RMS-normalized random activations times a real layer's QK-norm gammas (so the
score range matches that layer); v is random. Error is vs an fp32 CPU reference.
Usage: python3 attn_bench.py [variant ...]   (default: all)
Env: LAYER (default 25), REPS (default 20), L (default 7854), LK_CROSS (512)
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
B, N, D, DIM, TP = 2, 3, 128, 3072, 8
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


VARIANTS = {"sdpa": v_sdpa, "constmax": v_constmax, "cte": v_cte, "nst": v_nst,
            "x_sdpa": v_sdpa, "x_cte": v_cte}


def bench(name, fn, q, k, v, ref):
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
    flops = 4 * B * N * q.shape[1] * k.shape[1] * D
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
            bench(n, VARIANTS[n], q, k, v, ref)
    if cross_names:
        q, k, v = make_qkv(LAYER, L, LK_CROSS, kind="cross_attn")
        ref = ref_attn(q, k, v)
        print(f"cross-attn layer {LAYER}: B*n={B * N} Lq={L} Lk={LK_CROSS} d={D}", flush=True)
        for n in cross_names:
            bench(n, VARIANTS[n], q, k, v, ref)


if __name__ == "__main__":
    main()
