"""QKV projection of the SP DiT block on one NeuronCore (CFG-parallel TP4: this rank's 6 heads, all
tokens after the all-gather): the compiler's three F.linear calls (today) vs one fused F.linear vs
the nkilib qkv kernel. Values vs an fp32 CPU reference.
NEURON_RT_VISIBLE_CORES=<core> python3 scripts/validate/qkv_bench.py   Env: S (7856), REPS (20)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
S = int(os.environ.get("S", "7856"))
REPS = int(os.environ.get("REPS", "20"))
H, I1 = 3072, 768  # hidden, per-projection local width (6 heads x 128)


def three(x, wq, bq, wk, bk, wv, bv):
    return F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv)


def fused(x, w, b):
    return F.linear(x, w, b).split(I1, dim=-1)


def nki_qkv(x, w_hi, b):
    from nkilib.core.qkv.qkv import qkv
    y = qkv(x.unsqueeze(0), w_hi, bias=b)
    y = y[0] if isinstance(y, (list, tuple)) else y
    return y.reshape(-1, 3 * I1).split(I1, dim=-1)


def timeit(f, *a):
    f(*a)
    torch.neuron.synchronize()
    for _ in range(3):
        f(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    torch.neuron.set_device(0)
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(S, H, generator=g).to(torch.bfloat16)
    ws = [(torch.randn(I1, H, generator=g) * H ** -0.5).to(torch.bfloat16) for _ in range(3)]
    bs = [(0.1 * torch.randn(I1, generator=g)).to(torch.bfloat16) for _ in range(3)]
    ref = [F.linear(x.float(), w.float(), b.float()) for w, b in zip(ws, bs)]
    xd = x.to(dev)
    wd, bd = [w.to(dev) for w in ws], [b.to(dev) for b in bs]
    w_cat, b_cat = torch.cat(ws, 0), torch.cat(bs, 0)
    w_hi = w_cat.t().contiguous()  # [H, 3*I1] for the kernel (transposed on the CPU)
    kw = dict(backend="neuron", dynamic=False)
    flops = 2 * S * H * 3 * I1
    cases = [
        ("three F.linear", torch.compile(three, **kw), (xd, wd[0], bd[0], wd[1], bd[1], wd[2], bd[2])),
        ("fused F.linear", torch.compile(fused, **kw), (xd, w_cat.to(dev), b_cat.to(dev))),
        ("nkilib qkv", torch.compile(nki_qkv, **kw), (xd, w_hi.to(dev), b_cat.view(1, -1).to(dev))),
    ]
    for name, f, args in cases:
        try:
            ms, out = timeit(f, *args)
        except Exception as ex:  # noqa: BLE001
            print(f"{name:>15}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            continue
        rel = max(((o.float().cpu() - r).norm() / r.norm()).item() for o, r in zip(out, ref))
        print(f"{name:>15}: {ms:.3f} ms  {flops / ms / 1e9:.1f} TFLOP/s  max rel err {rel:.2e}", flush=True)


if __name__ == "__main__":
    main()
