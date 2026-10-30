"""CPU simulation (nki.simulate, no device) of kernels/qk_prep.py against the torch reference of
models/fast_dit (_tp_rms_qk without the all-reduce, _rope, _cte's permutes), at a small shape with
a partial last token tile. TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 qk_prep_sim.py [N (300)]
"""
import os
import sys

import nki
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.qk_prep import qk_apply_kernel, qk_sumsq  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
Np, H, D = N + 2, 6, 128
G, EPS = 3072, 1e-6
bf16 = torch.bfloat16


def rope(x, cos, sin):  # models.fast_dit._rope
    B, L, n, d = x.shape
    xp = x.float().reshape(B, L, n, d // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)
    return out.reshape(B, L, n, d).to(x.dtype)


def main():
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(Np, H * D, generator=g) * 3).to(bf16)
    k = (torch.randn(Np, H * D, generator=g) * 3).to(bf16)
    v = torch.randn(Np, H * D, generator=g).to(bf16)
    wq = (1 + 0.2 * torch.randn(H * D, generator=g)).to(bf16)
    wk = (1 + 0.2 * torch.randn(H * D, generator=g)).to(bf16)
    ang = torch.rand(N, D // 2, generator=g) * 50
    cos, sin = torch.cos(ang), torch.sin(ang)

    qf, kf = q.float(), k.float()
    ssq, ssk = qf.pow(2).sum(-1, keepdim=True), kf.pow(2).sum(-1, keepdim=True)
    qq = (qf * torch.rsqrt(ssq / G + EPS)).to(bf16) * wq
    kk = (kf * torch.rsqrt(ssk / G + EPS)).to(bf16) * wk
    c4, s4 = cos.view(1, N, 1, D // 2), sin.view(1, N, 1, D // 2)
    qq = rope(qq[:N].reshape(1, N, H, D), c4, s4)
    kk = rope(kk[:N].reshape(1, N, H, D), c4, s4)
    ref = (qq.permute(0, 2, 1, 3).reshape(H, N, D), kk.permute(0, 2, 3, 1).reshape(H, D, N),
           v[:N].reshape(1, N, H, D).permute(0, 2, 1, 3).reshape(H, N, D))

    ss = nki.simulate(qk_sumsq)(q, k)
    ss = torch.as_tensor(ss)
    sref = torch.cat([ssq, ssk], 1)
    print(f"[qk_prep_sim] N={N}: ss {tuple(ss.shape)} max rel diff "
          f"{((ss.float() - sref).abs() / sref).max().item():.2e}", flush=True)
    out = nki.simulate(qk_apply_kernel(D, EPS, G))(q, k, v, sref, wq, wk, cos, sin)
    qkv = torch.cat([q, k, v], 1)  # FAST_QKV_CTE: one [Np, 3D] projection output
    fused = nki.simulate(qk_apply_kernel(D, EPS, G, fused=True))(qkv, sref, wq, wk, cos, sin)
    for name, a, b in zip(("fused qn", "fused kn", "fused vn"), out, fused):
        print(f"[qk_prep_sim] {name} vs unfused: bitwise equal "
              f"{torch.equal(torch.as_tensor(a), torch.as_tensor(b))}", flush=True)
    for name, a, b in zip(("qn", "kn", "vn"), ref, out):
        b = torch.as_tensor(b)
        assert tuple(b.shape) == tuple(a.shape), (name, b.shape, a.shape)
        a, b = a.float(), b.float()
        dif = (a - b).abs()
        print(f"[qk_prep_sim] {name} {tuple(b.shape)}: max |diff| {dif.max().item():.3e}, "
              f"differing {(dif > 0).float().mean().item() * 100:.3f}%, rel rms "
              f"{(dif.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.2e}", flush=True)


if __name__ == "__main__":
    main()
