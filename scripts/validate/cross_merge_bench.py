"""Cross-attention with the zero-padded context keys merged (one NeuronCore, local cross shape).

The T5 context is zero-padded to 512 tokens and the DiT has no context mask, so every padded
position has the same text embedding, hence the same K and V in every layer. Attention over m
identical keys equals attention over one copy of that key with log(m) added to its score, so the
512 keys can be replaced by the n real ones plus copies of the pad key (each carrying
log(m / copies)), padded to LK keys: exact up to rounding. This bench times the 512-key kernel
the DiT uses today against merged-key variants and scores all of them against an fp32 CPU
reference on the full 512 keys.
  full_cte   attention_cte on all 512 keys (today)
  m_cte      attention_cte on LK keys with the log-weights as position_bias (q padded to 128s)
  m_cte_nb   attention_cte on LK keys, no bias (timing only: wrong weights)
  m_mm       bmm with fp32 scores + bias, fp32 softmax, bmm
  m_sdpa     F.scaled_dot_product_attention with the bias as a bf16 attn_mask
  m_sdpa32   the same with an fp32 attn_mask
python3 cross_merge_bench.py [variant ...]   Env: LAYER (25), REPS (20), LQ (1964), N (24 heads),
NREAL (70 cond / 126 uncond), LK (128)
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
LQ = int(os.environ.get("LQ", "1964"))
N = int(os.environ.get("N", "24"))
NREAL = int(os.environ.get("NREAL", "70"))
LK = int(os.environ.get("LK", "128"))
LFULL, D, DIM = 512, 128, 3072
dev = torch.device("neuron")


def gammas(layer):
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "diffusion_pytorch_model*.safetensors"))):
        with safe_open(f, "pt") as st:
            for n in (f"blocks.{layer}.cross_attn.norm_q.weight", f"blocks.{layer}.cross_attn.norm_k.weight"):
                if n in st.keys():
                    out[n.split(".")[-2]] = st.get_tensor(n).float()
    return out["norm_q"], out["norm_k"]


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def merged(k_real, k_pad, v_real, v_pad):
    """[1, n, N, D] real + [1, 1, N, D] pad -> LK keys/values and the per-key log-weight [LK]."""
    n = k_real.shape[1]
    copies = LK - n
    m = LFULL - n
    k = torch.cat([k_real, k_pad.expand(1, copies, N, D)], dim=1)
    v = torch.cat([v_real, v_pad.expand(1, copies, N, D)], dim=1)
    w = torch.zeros(LK)
    w[n:] = math.log(m / copies)
    return k, v, w


def ref_attn(q, k, v, w=None):  # q [1, Lq, N, D], k/v [1, Lk, N, D], w [Lk] -> [1, Lq, N, D]
    s = torch.einsum("qhd,khd->hqk", q[0].double(), k[0].double()) / math.sqrt(D)
    if w is not None:
        s = s + w.double()
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v[0].double()).unsqueeze(0).float()


def _heads(q, k, v):
    b, lq, n, d = q.shape
    lk = k.shape[1]
    return (q.permute(0, 2, 1, 3).reshape(b * n, lq, d), k.permute(0, 2, 3, 1).reshape(b * n, d, lk),
            v.permute(0, 2, 1, 3).reshape(b * n, lk, d))


def full_cte(q, k, v, w):
    from nkilib.core.attention.attention_cte import attention_cte
    qn, kn, vn = _heads(q, k, v)
    out = attention_cte(qn, kn, vn, scale=D ** -0.5, causal_mask=False)
    return out.view(1, N, -1, D).transpose(1, 2)


def _cte_bias(use_bias):
    def f(q, k, v, w):
        from nkilib.core.attention.attention_cte import attention_cte
        lq = q.shape[1]
        lp = -(-lq // 128) * 128
        qn, kn, vn = _heads(F.pad(q, (0, 0, 0, 0, 0, lp - lq)), k, v)
        kw = {}
        if use_bias:  # the kernel adds the bias before scaling: pre-divide
            kw["position_bias"] = (w.float() / D ** -0.5).view(1, 1, -1).expand(N, lp, -1).contiguous()
        out = attention_cte(qn, kn, vn, scale=D ** -0.5, causal_mask=False, **kw)
        return out.view(1, N, lp, D)[:, :, :lq].transpose(1, 2)
    return f


def m_mm(q, k, v, w):
    qh, kh, vh = _heads(q, k, v)
    s = torch.bmm(qh, kh, out_dtype=torch.float32) * D ** -0.5 + w.float().view(1, 1, -1)
    p = s.softmax(-1).to(q.dtype)
    return torch.bmm(p, vh).view(1, N, -1, D).transpose(1, 2)


def m_sdpa(q, k, v, w):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         attn_mask=w.to(q.dtype).view(1, 1, 1, -1))
    return out.transpose(1, 2)


def m_sdpa32(q, k, v, w):  # fp32 mask: the log-weights are not rounded to bf16
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         attn_mask=w.float().view(1, 1, 1, -1))
    return out.transpose(1, 2)


VARIANTS = {"full_cte": full_cte, "m_cte": _cte_bias(True), "m_cte_nb": _cte_bias(False),
            "m_mm": m_mm, "m_sdpa": m_sdpa, "m_sdpa32": m_sdpa32}


def main():
    names = sys.argv[1:] or list(VARIANTS)
    torch.set_grad_enabled(False)
    g = torch.Generator().manual_seed(0)
    gq, gk = gammas(LAYER)
    q = rms(torch.randn(1, LQ, DIM, generator=g), gq)[..., :N * D].reshape(1, LQ, N, D)
    kr = rms(torch.randn(1, NREAL + 1, DIM, generator=g), gk)[..., :N * D].reshape(1, NREAL + 1, N, D)
    vr = torch.randn(1, NREAL + 1, N, D, generator=g)
    k_real, k_pad, v_real, v_pad = kr[:, :NREAL], kr[:, NREAL:], vr[:, :NREAL], vr[:, NREAL:]
    # bf16 inputs, as on the device; the reference uses the same rounded values
    q, k_real, k_pad, v_real, v_pad = (t.to(torch.bfloat16).float() for t in (q, k_real, k_pad, v_real, v_pad))
    k_full = torch.cat([k_real, k_pad.expand(1, LFULL - NREAL, N, D)], dim=1)
    v_full = torch.cat([v_real, v_pad.expand(1, LFULL - NREAL, N, D)], dim=1)
    k_m, v_m, w_m = merged(k_real, k_pad, v_real, v_pad)
    ref = ref_attn(q, k_full, v_full)
    ref_m = ref_attn(q, k_m, v_m, w_m)
    exact = ((ref_m - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"cross layer {LAYER}: {N} heads x {LQ} queries; {NREAL} real keys + {LFULL - NREAL} pad -> {LK} "
          f"merged keys; merged vs full in fp64: rel-RMS {exact:.2e}", flush=True)
    for name in names:
        fn = VARIANTS[name]
        kk, vv, ww = (k_full, v_full, torch.zeros(LFULL)) if name.startswith("full") else (k_m, v_m, w_m)
        f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
        args = [t.to(torch.bfloat16).to(dev) for t in (q, kk, vv)] + [ww.to(dev)]
        t0 = time.perf_counter()
        try:
            out = f(*args)
            torch.neuron.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"{name:>9}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        t_first = time.perf_counter() - t0
        for _ in range(3):
            f(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = f(*args)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        o = out.float().cpu()
        rel = ((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"{name:>9}: {ms:7.3f} ms/call  rel_rms vs fp64 full {rel:.5f}  max_abs {(o - ref).abs().max().item():.4f}"
              f"  (first call {t_first:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
