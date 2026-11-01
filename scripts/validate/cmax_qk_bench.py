"""FAST_CMAX_QK's kernels/attention_cmax.attention_cmax_qk_kernel (QK-norm + RoPE prepared inside the
attention pipeline) against the two kernels it replaces, kernels/qk_prep's qk_apply (fused qkv,
q_t) followed by kernels/attention_cmax, at this pipeline's self-attention call (one TP rank: 6 heads
x 7854 tokens x 128, qkv [7856, 2304], one core). Prints time per call of each (timed twice,
interleaved) and the max |difference| of the outputs (same per-element operations, so 0 is
expected). NEURON_RT_VISIBLE_CORES=<core> python3 cmax_qk_bench.py   Env: REPS (20), SMAX (10.0)
"""
import math
import os
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
SMAX = float(os.environ.get("SMAX", "10.0"))
H, N, NP, D, G, EPS = 6, 7854, 7856, 128, 3072, 1e-6
bf16 = torch.bfloat16
def timed(f, args):
    out = f(*args)
    torch.neuron.synchronize()
    for _ in range(3):
        out = f(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*args)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out.cpu()
def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    HD = H * D
    qkv = torch.randn(NP, 3 * HD, generator=g).to(bf16)
    qkv[N:] = 0
    ss = torch.stack([qkv[:, :HD].float().pow(2).sum(-1), qkv[:, HD:2 * HD].float().pow(2).sum(-1)], -1) * (G / HD)
    wq = (1 + 0.1 * torch.randn(HD, generator=g)).to(bf16)
    wk = (1 + 0.1 * torch.randn(HD, generator=g)).to(bf16)
    ang = torch.rand(N, D // 2, generator=g) * 2 * math.pi
    cos, sin = ang.cos().float(), ang.sin().float()
    args = [t.to(dev) for t in (qkv, ss, wq, wk, cos, sin)]
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    from kernels.qk_prep import qk_apply_kernel
    from kernels.attention_cmax import attention_cmax, attention_cmax_qk_kernel
    k_qk = qk_apply_kernel(D, EPS, G, fused=True, q_t=True)
    k_fused = attention_cmax_qk_kernel(EPS, G)
    fns = {"qk_apply+cmax": torch.compile(lambda *a: attention_cmax(*k_qk(*a), SMAX, D ** -0.5), **kw),
           "fused": torch.compile(lambda *a: k_fused(*a, SMAX, D ** -0.5), **kw)}
    times = {nm: [] for nm in fns}
    outs = {}
    for _ in range(2):
        for nm, f in fns.items():
            t, outs[nm] = timed(f, args)
            times[nm].append(t)
    a, b = outs["qk_apply+cmax"].float(), outs["fused"].float()
    diff = (a - b).abs().max().item()
    print(f"[cmax_qk] {H} x {N} x {D} one core, max {SMAX}: qk_apply + attention_cmax "
          f"{times['qk_apply+cmax'][0]:.3f} / {times['qk_apply+cmax'][1]:.3f} ms, fused {times['fused'][0]:.3f} / "
          f"{times['fused'][1]:.3f} ms; max |fused - two kernels| {diff:.2e} (out rms {a.pow(2).mean().sqrt():.3e}), "
          f"finite {bool(torch.isfinite(b).all())}", flush=True)
if __name__ == "__main__":
    main()
