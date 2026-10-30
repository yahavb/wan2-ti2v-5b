"""Self-attention q / k / v projections of the SP block on one core ([Np, 3072] -> 3 x [Np, 768], with
bias, plus the per-token q / k sums of squares of FAST_QK_PREP): the compiler's three matmuls and
kernels/qk_prep.qk_sumsq vs nkilib's qkv_cte on the fused [3072, 2304] weight, which also writes the
sums while each tile is in SBUF. Prints time per call and the differences.
NEURON_RT_VISIBLE_CORES=<one core> python3 qkv_bench.py   Env: REPS (20)
"""
import os
import sys
import time

import nki
import nki.language as nl
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.qk_prep import qk_sumsq  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
NP, H, D, NH, DH = 7856, 3072, 768, 6, 128
bf16 = torch.bfloat16


@nki.jit
def qkv_ss(x, w, b):
    from nkilib.core.qkv.qkv_cte import qkv_cte
    from nkilib.core.utils.common_types import QKVOutputLayout
    B, S, _ = x.shape
    qss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    kss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    out = qkv_cte(input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.BSD, bias=b, d_head=DH,
                  num_q_heads=NH, num_kv_heads=NH, q_squared_sum_out=qss, k_squared_sum_out=kss)
    return out, qss, kss


def ref(x, wq, bq, wk, bk, wv, bv):
    q, k, v = F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)
    return q, k, v, qk_sumsq(q, k)


def new(x, w, b):
    return qkv_ss(x.unsqueeze(0), w, b)


def timed(f, args):
    out = f(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*args)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item()


@nki.jit
def oproj(a, w, b):
    from nkilib.core.output_projection.output_projection_cte.output_projection_cte import output_projection_cte
    from nkilib.core.utils.common_types import OProjAttentionLayout
    return output_projection_cte(a, w, bias=b, attention_layout=OProjAttentionLayout.BSNd)


def oproj_bench(dev, g):
    """The A2A_O / cross-attention output projection on the token shard: [S, 3072] x [3072, 3072] + bias."""
    S, NH2 = 1964, 24
    a = torch.randn(S, H, generator=g).to(bf16)
    w = (torch.randn(H, H, generator=g) * H ** -0.5).to(bf16)  # nn.Linear layout [out, in]
    b = (0.1 * torch.randn(H, generator=g)).to(bf16)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    t_ref, r = timed(torch.compile(lambda a_, w_, b_: F.linear(a_, w_, b_), **kw), [a.to(dev), w.to(dev), b.to(dev)])
    wt, bb = w.t().contiguous().to(dev), b.view(1, -1).to(dev)
    t_new, o = timed(torch.compile(lambda a_, w_, b_: oproj(a_.view(1, S, NH2, DH), w_, b_), **kw), [a.to(dev), wt, bb])
    print(f"[qkv_bench] O-projection [{S}, {H}] x [{H}, {H}]: compiler {t_ref:.3f} ms, nkilib output_projection_cte "
          f"{t_new:.3f} ms; rel rms {rel(r, o.reshape(S, H)):.2e}", flush=True)


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    if os.environ.get("OPROJ", "1") == "1":
        try:
            oproj_bench(dev, g)
        except Exception as ex:  # report and go on to the QKV case
            print(f"[qkv_bench] O-projection FAILED {type(ex).__name__}: {str(ex)[:400]}", flush=True)
    x = torch.randn(NP, H, generator=g).to(bf16)
    ws = [(torch.randn(D, H, generator=g) * H ** -0.5).to(bf16) for _ in range(3)]
    bs = [(0.1 * torch.randn(D, generator=g)).to(bf16) for _ in range(3)]
    w = torch.cat(ws, 0).t().contiguous()  # [H, 3D]
    b = torch.cat(bs, 0).view(1, -1)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    to = lambda t: t.to(dev)  # noqa: E731
    t_ref, (q, k, v, ss) = timed(torch.compile(ref, **kw), [to(x)] + [to(t) for p in zip(ws, bs) for t in p])
    t_new, (o, qss, kss) = timed(torch.compile(new, **kw), [to(x), to(w), to(b)])
    print(f"[qkv_bench] compiler 3 matmuls + qk_sumsq {t_ref:.3f} ms, nkilib qkv_cte + sums {t_new:.3f} ms "
          f"(one core, {REPS} reps)", flush=True)
    o = o[0].float().cpu()
    print(f"[qkv_bench] rel rms q {rel(q, o[:, :D]):.2e} k {rel(k, o[:, D:2 * D]):.2e} v {rel(v, o[:, 2 * D:]):.2e}; "
          f"sums q {rel(ss[:, 0], qss[0, :, 0]):.2e} k {rel(ss[:, 1], kss[0, :, 0]):.2e}", flush=True)


if __name__ == "__main__":
    main()
