"""FAST_QK_PREP kernels (kernels/qk_prep.py) vs the torch reference (models/fast_dit._tp_rms_qk +
_rope + _cte's permutes, without the all-reduce) on one core, at the SP block shape (N = 7854
tokens padded to Np = 7856, 6 local heads x 128) with the real RoPE tables. Prints the mismatch
against the reference and the time of each path as one compiled graph.
NEURON_RT_VISIBLE_CORES=<one core> python3 qk_prep_check.py   Env: REPS (20)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
import models.fast_dit as fd  # noqa: E402
from kernels.qk_prep import qk_apply_kernel, qk_sumsq  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
GRID = (21, 22, 17)
N = GRID[0] * GRID[1] * GRID[2]
Np = N + 2
H, D = 6, 128
G, EPS = 3072, 1e-6
bf16 = torch.bfloat16


def rope_freqs(max_seq_len, dim, theta=10000):
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    return torch.outer(torch.arange(max_seq_len), freqs).float()


def ref(q, k, v, cos, sin, wq, wk):
    qf, kf = q.float(), k.float()
    qq = (qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) / G + EPS)).type_as(q) * wq
    kk = (kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) / G + EPS)).type_as(k) * wk
    qq = fd._rope(qq[:N].reshape(1, N, H, D), cos, sin)
    kk = fd._rope(kk[:N].reshape(1, N, H, D), cos, sin)
    vv = v[:N].reshape(1, N, H, D)
    return (qq.permute(0, 2, 1, 3).reshape(H, N, D), kk.permute(0, 2, 3, 1).reshape(H, D, N),
            vv.permute(0, 2, 1, 3).reshape(H, N, D))


def new(q, k, v, cos, sin, wq, wk, gp=False):
    ss = qk_sumsq(q, k)
    return qk_apply_kernel(D, EPS, G, gp)(q, k, v, ss, wq, wk, cos.view(N, D // 2), sin.view(N, D // 2))


def new_gp(q, k, v, cos, sin, wq, wk):
    return new(q, k, v, cos, sin, wq, wk, gp=True)


def timed(f, args):
    f(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*args)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(Np, H * D, generator=g) * 3).to(bf16).to(dev)
    k = (torch.randn(Np, H * D, generator=g) * 3).to(bf16).to(dev)
    v = torch.randn(Np, H * D, generator=g).to(bf16).to(dev)
    wq = (1 + 0.2 * torch.randn(H * D, generator=g)).to(bf16).to(dev)
    wk = (1 + 0.2 * torch.randn(H * D, generator=g)).to(bf16).to(dev)
    c = D // 2
    freqs = torch.cat([rope_freqs(1024, D - 4 * (D // 6)), rope_freqs(1024, 2 * (D // 6)),
                       rope_freqs(1024, 2 * (D // 6))], 1)
    cos, sin = fd._rope_tables(freqs, GRID, D, dev)  # [1, N, 1, 64] fp32
    assert cos.shape == (1, N, 1, c), cos.shape
    args = (q, k, v, cos, sin, wq, wk)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    t_ref, r = timed(torch.compile(ref, **kw), args)
    t_new, o = timed(torch.compile(new, **kw), args)
    t_gp, o_gp = timed(torch.compile(new_gp, **kw), args)
    print(f"[qk_prep] torch reference {t_ref:.3f} ms, kernels {t_new:.3f} ms, kernels with the GpSimd split "
          f"{t_gp:.3f} ms (one core, {REPS} reps)", flush=True)
    for a, b in zip(o, o_gp):
        assert torch.equal(a.cpu(), b.cpu()), "GpSimd split changed the result"
    for name, a, b in zip(("qn", "kn", "vn"), r, o):
        a, b = a.float().cpu(), b.float().cpu()
        dif = (a - b).abs()
        rel = (dif.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item()
        ulp = (dif / a.abs().clamp(min=1e-30)).max().item()  # bf16 ulp ~ 2^-8 relative
        print(f"[qk_prep] {name} {tuple(b.shape)}: max |diff| {dif.max().item():.3e}, "
              f"differing {(dif > 0).float().mean().item() * 100:.3f}%, rel rms {rel:.2e}, max rel {ulp:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
