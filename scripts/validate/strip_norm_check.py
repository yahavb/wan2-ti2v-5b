"""kernels/vae_strip_norm.py vs the compiled torch strip norm graph (one core): values and time,
at the strip shapes of the 704x544 decode (VAE_TCHUNK=2), including an image-border strip.
NEURON_RT_VISIBLE_CORES=<core> python3 scripts/validate/strip_norm_check.py   Env: REPS (20)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.vae_strip_norm import strip_norm_cat  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
CASES = [  # C, T, h, W, k (rows sent per rank), ia, ib
    (256, 8, 44, 272, 2, 5, 8),
    (512, 8, 44, 272, 2, 5, 8),
    (512, 8, 22, 136, 2, -1, 2),
    (1024, 4, 11, 68, 2, 13, -1),
    (1024, 2, 6, 34, 4, 7, 13),
    (256, 1, 44, 272, 2, 5, 8),
]
WORLD = 8


def rms_silu(x, gamma):
    rms = x.float().pow(2).mean(1, keepdim=True).clamp(min=1e-8).rsqrt()
    return F.silu((x * rms).to(gamma.dtype) * gamma)


def row(g, i):
    if i < 0:
        return g.new_zeros(1, g.shape[1], g.shape[2], 1, g.shape[4])
    return g[i:i + 1]


def ref(x, gamma, prev, g, ia: int, ib: int):
    y = torch.cat([row(g, ia), rms_silu(x, gamma), row(g, ib)], dim=3)
    z = torch.cat([prev, y], dim=2)
    return z, z[:, :, -2:].clone()


def timeit(f):
    f()
    torch.neuron.synchronize()
    for _ in range(3):
        f()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    torch.neuron.set_device(0)
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    f_ref = torch.compile(ref, backend="neuron", dynamic=False)
    for C, T, h, W, k, ia, ib in CASES:
        g0 = torch.Generator().manual_seed(0)
        x = torch.randn(1, C, T, h, W, generator=g0).to(torch.bfloat16).to(dev)
        gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=g0)).to(torch.bfloat16).to(dev)
        prev = torch.randn(1, C, 2, h + 2, W, generator=g0).to(torch.bfloat16).to(dev)
        g = torch.randn(WORLD * k, C, T, 1, W, generator=g0).to(torch.bfloat16).to(dev)
        kern = strip_norm_cat(T, h, W, ia, ib)

        def run_k():
            z2, c2 = kern(x.view(C, T * h * W), gamma.view(C, 1), prev.view(C, 2 * (h + 2) * W),
                          g.view(WORLD * k * C, T * W))
            return z2.view(1, C, T + 2, h + 2, W), c2.view(1, C, 2, h + 2, W)

        tr, (zr, cr) = timeit(lambda: f_ref(x, gamma, prev, g, ia, ib))
        tk, (zk, ck) = timeit(run_k)
        zr, cr, zk, ck = (a.float().cpu() for a in (zr, cr, zk, ck))
        dz = (zk - zr).abs()
        rel = ((zk - zr).norm() / zr.norm()).item()
        big = (dz > 0.02 * zr.abs().clamp(min=0.5)).float().mean().item()
        cok = torch.equal(ck, zk[:, :, -2:])
        print(f"C={C} T={T} {h}x{W} ia={ia} ib={ib}: torch {tr:.3f} ms, nki {tk:.3f} ms; "
              f"z max|d| {dz.max().item():.4f} rel-RMS {rel:.2e} >2%: {big * 100:.3f}%; "
              f"cache == z[-2:] {cok}; halo rows equal {torch.equal(zk[:, :, 2:, [0, -1]], zr[:, :, 2:, [0, -1]])}; "
              f"prev equal {torch.equal(zk[:, :, :2], zr[:, :, :2])}", flush=True)


if __name__ == "__main__":
    main()
