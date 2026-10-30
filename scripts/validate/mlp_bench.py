"""The SP block's FFN on one core at this pipeline's shape (FAST_NKI_MLP: nkilib mlp, up + bias,
GELU-tanh, down, on the all-gathered [7856, 3072] tokens with this rank's [3072, 3584] / [3584, 3072]
weight shards) vs the compiler's two matmuls, with the achieved TFLOP/s. Set PROFILE_DIR to capture
a device profile of the kernel (NEURON_RT_INSPECT_*).
NEURON_RT_VISIBLE_CORES=<one core> python3 mlp_bench.py   Env: REPS (20)
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

REPS = int(os.environ.get("REPS", "20"))
NP, H, I = 7856, 3072, 3584
bf16 = torch.bfloat16
PEAK_TFLOPS = 83.2  # bf16 dense, one trn2 NeuronCore


def nki_mlp(x, w_up, b_up, w_down):
    from nkilib.core.mlp.mlp import mlp
    from nkilib.core.utils.common_types import ActFnType
    y = mlp(x.unsqueeze(0), w_up, w_up, w_down, up_proj_bias_tensor=b_up,
            activation_fn=ActFnType.GELU_Tanh_Approx, skip_gate_proj=True)
    return (y[0] if isinstance(y, (list, tuple)) else y).reshape(NP, H)


def torch_mlp(x, w0, b0, w2):
    return F.linear(F.gelu(F.linear(x, w0, b0), approximate="tanh"), w2)


def timed(f, args):
    out = f(*args)
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
    x = torch.randn(NP, H, generator=g).to(bf16)
    w0 = (torch.randn(I, H, generator=g) * H ** -0.5).to(bf16)  # nn.Linear layouts
    b0 = (0.1 * torch.randn(I, generator=g)).to(bf16)
    w2 = (torch.randn(H, I, generator=g) * I ** -0.5).to(bf16)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    flop = 2 * NP * H * I * 2
    t_ref, r = timed(torch.compile(torch_mlp, **kw), [x.to(dev), w0.to(dev), b0.to(dev), w2.to(dev)])
    args = [x.to(dev), w0.t().contiguous().to(dev), b0.view(1, -1).to(dev), w2.t().contiguous().to(dev)]
    t_new, o = timed(torch.compile(nki_mlp, **kw), args)
    a, b = r.float().cpu(), o.float().cpu()
    print(f"[mlp_bench] compiler {t_ref:.3f} ms ({flop / t_ref / 1e9:.1f} TFLOP/s, "
          f"{flop / t_ref / 1e9 / PEAK_TFLOPS * 100:.0f}% of peak), nkilib mlp {t_new:.3f} ms "
          f"({flop / t_new / 1e9:.1f} TFLOP/s, {flop / t_new / 1e9 / PEAK_TFLOPS * 100:.0f}%); "
          f"rel rms {((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.2e}", flush=True)


if __name__ == "__main__":
    main()
