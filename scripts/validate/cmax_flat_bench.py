"""kernels/attention_cmax (FAST_CMAX_FLAT: one pipeline over all heads and query blocks, Q / K / V
resident in SBUF) against nkilib's experimental attention_const_max (one pipeline per query block,
K / V streamed per block), both with the 1e-30 softmax-sum floor, at this pipeline's self-attention
call (6 heads x 7854 tokens x 128, q / k head-major, one core). Prints time per call (each kernel
timed twice, interleaved), the max |difference| between the two outputs (same instructions per KV
tile, so 0 is expected) and nkilib's rel-RMS against an fp32 CPU softmax on head 0.
NEURON_RT_VISIBLE_CORES=<core> python3 cmax_flat_bench.py   Env: REPS (20), SMAX (10.0)
"""
import os
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
SMAX = float(os.environ.get("SMAX", "10.0"))
BS, L, D = 6, 7854, 128
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
    q = torch.randn(BS, L, D, generator=g)
    k = torch.randn(BS, L, D, generator=g)
    q = (q / q.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16)
    k = (k / k.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16)
    v = torch.randn(BS, L, D, generator=g).to(bf16)
    s = q[0].float() @ k[0].float().t() * D ** -0.5
    ref = torch.softmax(s, -1) @ v[0].float()  # head 0, fp32
    rel = lambda o: ((o.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()  # noqa: E731
    args = [q.transpose(1, 2).contiguous().to(dev), k.transpose(1, 2).contiguous().to(dev), v.to(dev)]
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    import nkilib.experimental.attention.attention_const_max as m
    m._EPS = 1e-30  # as models/fast_dit._cmax_kernel
    from kernels.attention_cmax import attention_cmax
    fns = {"nkilib": torch.compile(lambda q_, k_, v_: m.attention_const_max(q_, k_, v_, SMAX, D ** -0.5), **kw),
           "flat": torch.compile(lambda q_, k_, v_: attention_cmax(q_, k_, v_, SMAX, D ** -0.5), **kw)}
    times = {nm: [] for nm in fns}
    outs = {}
    for _ in range(2):
        for nm, f in fns.items():
            t, outs[nm] = timed(f, args)
            times[nm].append(t)
    diff = (outs["flat"].float() - outs["nkilib"].float()).abs().max().item()
    print(f"[cmax_flat] {BS} x {L} x {D} one core, max {SMAX}: nkilib {times['nkilib'][0]:.3f} / "
          f"{times['nkilib'][1]:.3f} ms (rel rms {rel(outs['nkilib'][0]):.3e}), flat {times['flat'][0]:.3f} / "
          f"{times['flat'][1]:.3f} ms; max |flat - nkilib| {diff:.2e}, finite "
          f"{bool(torch.isfinite(outs['flat'].float()).all())}", flush=True)
if __name__ == "__main__":
    main()
