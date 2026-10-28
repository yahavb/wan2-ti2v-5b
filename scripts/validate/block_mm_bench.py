"""The SP block's matmuls at the current per-rank shapes (CFG-parallel TP4, batch 1, Np = 7856
padded tokens, shard S = 1964), on one core: all 8 in one compiled graph, then each alone.
F.linear with [out, in] weights, no bias, like models/fast_dit._mm. Peak = 83.4 TFLOP/s bf16.
Usage: python3 block_mm_bench.py
"""
import time

import torch
import torch.nn.functional as F

REPS = 20
PEAK = 83.4e12
Np, S, C, HL, FL = 7856, 1964, 3072, 768, 3584


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)

    def r(*s, sc=0.5):
        return (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).to(dev)

    hf, xs, ys, hh = r(Np, C), r(S, C), r(Np, HL), r(Np, FL)
    W = {"q": r(HL, C, sc=0.02), "k": r(HL, C, sc=0.02), "v": r(HL, C, sc=0.02), "o": r(C, HL, sc=0.02),
         "xq": r(C, C, sc=0.02), "xo": r(C, C, sc=0.02), "ffn1": r(FL, C, sc=0.02), "ffn2": r(C, FL, sc=0.02)}
    IN = {"q": hf, "k": hf, "v": hf, "o": ys, "xq": xs, "xo": xs, "ffn1": hf, "ffn2": hh}

    def flops(n):
        return 2 * IN[n].shape[0] * IN[n].shape[1] * W[n].shape[0]

    def bench(fn):
        f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
        f()
        torch.neuron.synchronize()
        for _ in range(3):
            f()
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f()
        torch.neuron.synchronize()
        return (time.perf_counter() - t0) / REPS * 1e3

    names = list(W)
    ms = bench(lambda: tuple(F.linear(IN[n], W[n]) for n in names))
    tot = sum(flops(n) for n in names)
    print(f"all 8 in one graph: {ms:.3f} ms, {tot / 1e9:.1f} GFLOP, {tot / ms / 1e9:.1f} TFLOP/s "
          f"({100 * tot / ms / 1e9 / (PEAK / 1e12):.0f}% of peak)", flush=True)
    for n in names:
        m = bench(lambda n=n: F.linear(IN[n], W[n]))
        print(f"  {n:>5} [{IN[n].shape[0]}, {IN[n].shape[1]}] x [{W[n].shape[1]}, {W[n].shape[0]}]: {m:.3f} ms "
              f"{flops(n) / m / 1e9:6.1f} TFLOP/s ({100 * flops(n) / m / 1e9 / (PEAK / 1e12):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
