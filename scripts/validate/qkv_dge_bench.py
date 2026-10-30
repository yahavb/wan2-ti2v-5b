"""nkilib qkv_cte at the SP block's shape ([1, 7856, 3072] @ [3072, 2304] + bias, with the q / k
sums of squares) with its software-DGE DMAs (each one ~0.75 us of GpSimd descriptor generation) as
they are, or switched to hardware DGE by patching the module's dge_mode before the kernel is traced.
One core:  NEURON_RT_VISIBLE_CORES=<core> python3 qkv_dge_bench.py
Env: DGE (sw | hw), REPS (50), OUT_DIR (/tmp/qkv_dge): outputs saved per DGE and compared bitwise.
"""
import os
import sys
import time

import nki
import nki.language as nl
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
DGE = os.environ.get("DGE", "sw")
REPS = int(os.environ.get("REPS", "50"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/qkv_dge")
NP, H, D, NH, DH = 7856, 3072, 768, 6, 128
bf16 = torch.bfloat16

if DGE == "hw":
    import nkilib.core.qkv.qkv_cte as _m
    from nki.isa.constants import dge_mode as _dm

    class _HW:  # every dge_mode.swdge in qkv_cte.py becomes hwdge; the other modes unchanged
        swdge = _dm.hwdge
        hwdge = _dm.hwdge
        none = _dm.none
    _m.dge_mode = _HW


def kernel(x, w, b):
    from nkilib.core.qkv.qkv_cte import qkv_cte
    from nkilib.core.utils.common_types import QKVOutputLayout
    B, S, _ = x.shape
    qss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    kss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    out = qkv_cte(input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.BSD, bias=b, d_head=DH,
                  num_q_heads=NH, num_kv_heads=NH, q_squared_sum_out=qss, k_squared_sum_out=kss)
    return out, qss, kss


kernel.__name__ = kernel.__qualname__ = f"qkv_dge_{DGE}"
qkv = nki.jit(kernel)


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, NP, H, generator=g).to(bf16).to(dev)
    w = (torch.randn(H, 3 * D, generator=g) * H ** -0.5).to(bf16).to(dev)
    b = (0.1 * torch.randn(1, 3 * D, generator=g)).to(bf16).to(dev)
    f = torch.compile(lambda x_, w_, b_: qkv(x_, w_, b_), backend="neuron", dynamic=False, fullgraph=True)
    out = f(x, w, b)
    torch.neuron.synchronize()
    for _ in range(3):
        out = f(x, w, b)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(x, w, b)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    o = [t.float().cpu() for t in out]
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(o, os.path.join(OUT_DIR, f"{DGE}.pt"))
    cmp = ""
    other = os.path.join(OUT_DIR, "sw.pt")
    if DGE != "sw" and os.path.exists(other):
        r = torch.load(other)
        cmp = "; vs sw: " + ", ".join(f"{n} bitwise {torch.equal(a, c)}" for n, a, c in zip(("qkv", "qss", "kss"), o, r))
    print(f"[qkv_dge] {DGE}: {ms:.3f} ms per call (one core, {REPS} reps){cmp}", flush=True)


if __name__ == "__main__":
    main()
