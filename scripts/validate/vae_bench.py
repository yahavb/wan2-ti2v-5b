"""Time the Wan2.2 VAE decode of one production tile on one NeuronCore, per decoder stage.

Decodes a [48, 21, TH, TW] latent (default 15x21 = the 8-rank tiled decode's tile) the same way
models/vae_tiled.py does (vae.decode -> model.decode, 21 causal chunks), with device syncs
around each top-level decoder stage, accumulated over the chunks.
Usage: python3 vae_bench.py   Env: TH (15), TW (21), REPS (2), UNTIMED_REPS (1), USE_NKI_CONV3D=1 ...
"""
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TH, TW = int(os.environ.get("TH", "15")), int(os.environ.get("TW", "21"))
REPS = int(os.environ.get("REPS", "2"))
T = int(os.environ.get("T_LAT", "21"))


def sync():
    torch.neuron.synchronize()


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    from wan.modules.vae2_2 import Wan2_2_VAE
    vae = Wan2_2_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.2_VAE.pth"), device=torch.device("cpu"))
    vae.model = vae.model.to(device=dev, dtype=torch.bfloat16)
    vae.scale = [s.to(device=dev, dtype=torch.bfloat16) for s in vae.scale]
    dec = vae.model.decoder
    acc = defaultdict(float)
    counts = defaultdict(int)
    timing = {"on": False}

    def wrap(name, mod):
        orig = mod.forward

        def fwd(*a, **k):
            if not timing["on"]:
                return orig(*a, **k)
            sync()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            sync()
            acc[name] += time.perf_counter() - t0
            counts[name] += 1
            return out
        mod.forward = fwd

    wrap("conv1", dec.conv1)
    for i, m in enumerate(dec.middle):
        wrap(f"middle.{i}.{type(m).__name__}", m)
    for i, m in enumerate(dec.upsamples):
        wrap(f"up.{i}", m)
    wrap("head", dec.head)

    g = torch.Generator().manual_seed(0)
    z = torch.randn(48, T, TH, TW, generator=g).to(torch.bfloat16).to(dev)
    for rep in range(REPS + 1):
        timing["on"] = rep > 0
        acc.clear()
        counts.clear()
        sync()
        t0 = time.perf_counter()
        out = vae.decode([z])[0]
        sync()
        tot = time.perf_counter() - t0
        tag = "warmup/compile" if rep == 0 else f"timed rep {rep}"
        print(f"[vae_bench] {tag}: decode {tot:.3f}s out {tuple(out.shape)}", flush=True)
        if rep > 0:
            s = sum(acc.values())
            for k, v in acc.items():
                print(f"    {k:<28} {v * 1e3:8.1f} ms  ({counts[k]} calls, {100 * v / s:5.1f}%)", flush=True)
            print(f"    {'sum of stages':<28} {s * 1e3:8.1f} ms; untimed remainder {1e3 * (tot - s):.1f} ms", flush=True)
    # untimed (no per-stage syncs) decodes, the production number (UNTIMED_REPS of them)
    for _ in range(int(os.environ.get("UNTIMED_REPS", "1"))):
        sync()
        t0 = time.perf_counter()
        out = vae.decode([z])[0]
        t_host = time.perf_counter() - t0  # host-side enqueue time (device runs async)
        sync()
        print(f"[vae_bench] untimed decode {time.perf_counter() - t0:.3f}s (host enqueue {t_host:.3f}s)", flush=True)
    ref = os.environ.get("VAE_REF_OUT")
    if ref:  # compare against a saved output of another configuration (exactness check)
        o = out.float().cpu()
        if os.path.exists(ref):
            r = torch.load(ref)
            print(f"[vae_bench] vs {ref}: max_abs {(o - r).abs().max().item():.5f} "
                  f"rel_rms {((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item():.6f}", flush=True)
        else:
            torch.save(o, ref)
            print(f"[vae_bench] saved {ref}", flush=True)


if __name__ == "__main__":
    main()
