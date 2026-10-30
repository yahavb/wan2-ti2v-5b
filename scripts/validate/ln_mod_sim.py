"""CPU simulation (nki.simulate, no device) of kernels/ln_mod.py against the torch code it replaces
in models/fast_dit._FastBlockSP (WanLayerNorm + modulation by one-hot rows; the affine norm3), at a
small token count with a partial last tile, a frame boundary inside a tile and pad tokens (all-zero
one-hot rows) at the end.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 ln_mod_sim.py [S (300)]
"""
import os
import sys

import nki
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.ln_mod import ln_affine_kernel, ln_mod_kernel  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 300
C, NF, EPS = 3072, 21, 1e-6
bf16 = torch.bfloat16


def report(name, a, b):
    b = torch.as_tensor(b)
    assert tuple(b.shape) == tuple(a.shape), (name, b.shape, a.shape)
    a, b = a.float(), b.float()
    dif = (a - b).abs()
    print(f"[ln_mod_sim] {name} {tuple(b.shape)}: max |diff| {dif.max().item():.3e}, differing "
          f"{(dif > 0).float().mean().item() * 100:.3f}%, rel rms "
          f"{(dif.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.2e}", flush=True)


def main():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(S, C, generator=g) * 4 + 0.5).to(bf16)
    emf = (torch.randn(NF, 6, C, generator=g) * 0.3).to(bf16)
    frame = (torch.arange(S) + 300) // 374  # a shard starting mid-frame: the boundary is in tile 0
    oh = F.one_hot(frame, NF).to(bf16)  # [S, F]
    oh[S - 7:] = 0  # pad tokens
    ohT = oh.t().contiguous()
    mods = [torch.matmul(oh, emf[:, j]) for j in range(6)]
    ln = lambda t: F.layer_norm(t, (C,), eps=EPS)  # noqa: E731 (WanLayerNorm on bf16)
    for js, jt in ((1, 0), (4, 3)):
        rows = (emf[:, js].contiguous(), emf[:, jt].contiguous())
        ref = ln(x) * (1 + mods[js]) + mods[jt]  # eager bf16: rounded after every op
        ref32 = (F.layer_norm(x.float(), (C,), eps=EPS) * (1 + mods[js].float()) + mods[jt].float()).to(bf16)
        out32 = nki.simulate(ln_mod_kernel(EPS))(x, ohT, *rows)
        report(f"ln_mod({js},{jt}) vs fp32 math", ref32, out32)
        report(f"ln_mod({js},{jt}) eager bf16 vs fp32 math", ref32, ref)
    w = (1 + 0.2 * torch.randn(C, generator=g)).to(bf16)
    b = (0.1 * torch.randn(C, generator=g)).to(bf16)
    ref = F.layer_norm(x, (C,), w, b, eps=EPS)
    report("ln_affine", ref, nki.simulate(ln_affine_kernel(EPS))(x, w, b))


if __name__ == "__main__":
    main()
