"""FAST_LN_MOD kernels (kernels/ln_mod.py) vs the torch code they replace in
models/fast_dit._FastBlockSP, on one core at the SP block's token shard ([1964, 3072] bf16, 21
frames): norm1 + modulation, and the affine norm3. Prints the mismatch and the time of each path as
one compiled graph. NEURON_RT_VISIBLE_CORES=<one core> python3 ln_mod_check.py   Env: REPS (20)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.ln_mod import ln_affine_kernel, ln_mod_kernel  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
S, C, NF, EPS = 1964, 3072, 21, 1e-6
bf16 = torch.bfloat16


def ref_mod(x, oh, emf):
    mods = [torch.matmul(oh, emf[:, j]) for j in range(2)]
    return F.layer_norm(x, (C,), eps=EPS) * (1 + mods[1]) + mods[0]


def new_mod(x, oh, emf):
    return ln_mod_kernel(EPS)(x, oh.t().contiguous(), emf[:, 1].contiguous(), emf[:, 0].contiguous())


def ref_aff(x, w, b):
    return F.layer_norm(x, (C,), w, b, eps=EPS)


def new_aff(x, w, b):
    return ln_affine_kernel(EPS)(x, w, b)


def timed(f, args):
    f(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*args)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def report(name, a, b):
    a, b = a.float().cpu(), b.float().cpu()
    dif = (a - b).abs()
    print(f"[ln_mod] {name}: max |diff| {dif.max().item():.3e}, differing {(dif > 0).float().mean().item() * 100:.3f}%, "
          f"rel rms {(dif.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.2e}", flush=True)


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(S, C, generator=g) * 4 + 0.5).to(bf16).to(dev)
    emf = (torch.randn(NF, 6, C, generator=g) * 0.3).to(bf16).to(dev)
    frame = (torch.arange(S) + 2 * 1964) // 374  # rank 2's shard of the 21 x 374 tokens
    oh = F.one_hot(frame, NF).to(bf16).to(dev)
    w = (1 + 0.2 * torch.randn(C, generator=g)).to(bf16).to(dev)
    b = (0.1 * torch.randn(C, generator=g)).to(bf16).to(dev)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    for name, ref, new, args in (("norm1+mod", ref_mod, new_mod, (x, oh, emf)), ("norm3 affine", ref_aff, new_aff, (x, w, b))):
        t_ref, r = timed(torch.compile(ref, **kw), args)
        t_new, o = timed(torch.compile(new, **kw), args)
        print(f"[ln_mod] {name}: torch {t_ref:.3f} ms, kernel {t_new:.3f} ms (one core, {REPS} reps)", flush=True)
        report(name + " kernel vs compiled torch", r, o)
        cpu = ref(*[a.cpu() for a in args])  # torch on the CPU: fp32 statistics, one rounding
        report(name + " compiled torch vs CPU torch", cpu, r)
        report(name + " kernel vs CPU torch", cpu, o)


if __name__ == "__main__":
    main()
