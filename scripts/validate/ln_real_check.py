"""FAST_LN_MOD kernels on a real residual-stream shard (the norm1 input of one block on the step-0
input, saved by an fp32 CPU forward) vs the same math in fp32 on the CPU, next to the compiled torch
code they replace. One core:  NEURON_RT_VISIBLE_CORES=<core> python3 ln_real_check.py <resid.pt>
The file holds x [L, C] bf16, em [F, 6, C] bf16 (modulation + time embedding), w3 / b3 [C] (norm3).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.ln_mod import ln_affine_kernel, ln_mod_kernel  # noqa: E402

EPS, S, TPF = 1e-6, 1964, 374  # shard rows, tokens per frame (22 x 17)
bf16 = torch.bfloat16


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    d = torch.load(sys.argv[1])
    em, w3, b3 = d["em"], d["w3"], d["b3"]
    nf, C = em.shape[0], em.shape[2]
    shard = int(os.environ.get("SHARD", "1"))
    x = d["x"][shard * S:(shard + 1) * S].contiguous()
    frame = (torch.arange(S) + shard * S) // TPF
    oh = F.one_hot(frame, nf).to(bf16)
    mods = [oh.float() @ em[:, j].float() for j in range(6)]
    ref1 = (F.layer_norm(x.float(), (C,), eps=EPS) * (1 + mods[1]) + mods[0])  # fp32 truth
    ref3 = F.layer_norm(x.float(), (C,), w3.float(), b3.float(), eps=EPS)
    xd, ohd, emd = x.to(dev), oh.to(dev), em.to(dev)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    torch_mod = torch.compile(lambda t, o, e: F.layer_norm(t, (C,), eps=EPS).to(bf16) * (1 + o @ e[:, 1]) + o @ e[:, 0], **kw)
    torch_aff = torch.compile(lambda t, w, b: F.layer_norm(t, (C,), w, b, eps=EPS), **kw)
    rows = (emd[:, 1].contiguous(), emd[:, 0].contiguous())
    k32 = torch.compile(lambda t, o, a, b: ln_mod_kernel(EPS)(t, o, a, b), **kw)
    ka = torch.compile(lambda t, w, b: ln_affine_kernel(EPS)(t, w, b), **kw)
    ohT = ohd.t().contiguous()
    print(f"[ln_real] x |max| {x.float().abs().max().item():.1f}; norm1+mod rel rms vs fp32: "
          f"compiled torch {rel(torch_mod(xd, ohd, emd), ref1):.3e}, kernel {rel(k32(xd, ohT, *rows), ref1):.3e}, "
          f"bf16(fp32 truth) {rel(ref1.to(bf16), ref1):.3e}", flush=True)
    print(f"[ln_real] norm3 affine rel rms vs fp32: compiled torch {rel(torch_aff(xd, w3.to(dev), b3.to(dev)), ref3):.3e}, "
          f"kernel {rel(ka(xd, w3.to(dev), b3.to(dev)), ref3):.3e}, bf16(fp32 truth) {rel(ref3.to(bf16), ref3):.3e}", flush=True)


if __name__ == "__main__":
    main()
