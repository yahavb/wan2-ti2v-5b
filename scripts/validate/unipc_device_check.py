"""DeviceFlowUniPC (models/device_unipc.py) on the device vs the upstream scheduler on the CPU, for
the pipeline's 10-step schedule, bf16 latents [1, 48, 21, 44, 34], and pseudo-random model outputs
that are identical for both. Prints per-step max abs difference and whether they are bit-identical;
also the same upstream scheduler run on the CPU twice (must be identical) and the device scheduler
fed CPU tensors (isolates the code change from device arithmetic).
Usage: python3 unipc_device_check.py   (one NeuronCore)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from models.device_unipc import DeviceFlowUniPC  # noqa: E402
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402

STEPS, SHIFT = 10, float(os.environ.get("SHIFT", "5.0"))
SHAPE = (1, 48, 21, 44, 34)


def run(cls, device):
    s = cls(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    s.set_timesteps(STEPS, device=torch.device("cpu"), shift=SHIFT)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(SHAPE, generator=g).to(torch.bfloat16).to(device)
    outs = []
    t0 = time.perf_counter()
    for i, t in enumerate(s.timesteps):
        mo = (torch.randn(SHAPE, generator=g) * 0.7 + 0.1 * i).to(torch.bfloat16).to(device)
        x = s.step(mo, t, x, return_dict=False)[0]
        outs.append(x)
    outs = [o.cpu() for o in outs]
    return outs, time.perf_counter() - t0


def diff(a, b):
    return [(float((p.float() - q.float()).abs().max()), bool(torch.equal(p, q))) for p, q in zip(a, b)]


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    ref, _ = run(FlowUniPCMultistepScheduler, torch.device("cpu"))
    ref2, _ = run(FlowUniPCMultistepScheduler, torch.device("cpu"))
    print(f"upstream CPU twice: {diff(ref, ref2)[-1]}")
    cpu_new, _ = run(DeviceFlowUniPC, torch.device("cpu"))
    d = diff(cpu_new, ref)
    print(f"DeviceFlowUniPC on CPU vs upstream CPU: all steps identical {all(x[1] for x in d)}; per step {d}")
    if os.environ.get("CPU_ONLY", "0") == "1":
        return
    run(DeviceFlowUniPC, dev)  # warm-up: compiles the eager device ops
    dev_new, secs = run(DeviceFlowUniPC, dev)
    print(f"DeviceFlowUniPC on device vs upstream CPU, per step (max abs, identical): {diff(dev_new, ref)}")
    rel = float((dev_new[-1].float() - ref[-1].float()).pow(2).mean().sqrt() / ref[-1].float().pow(2).mean().sqrt())
    print(f"final step rel-RMS device vs CPU {rel:.3e}; 10 device steps incl. host copy-back {1e3 * secs:.1f} ms")


if __name__ == "__main__":
    main()
