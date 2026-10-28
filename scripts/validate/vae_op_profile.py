"""Call sites of one Wan2.2 VAE tile decode on one NeuronCore: every conv, fused norm graph,
resample, shortcut and glue step, with its shape and number of calls per decode.

Every timed call is wrapped in device syncs; times are EXCLUSIVE (a call minus the timed calls
nested inside it), so the rows add up to the timed decode. USE IT FOR CALL SITES AND COUNTS, NOT
TIME: synced one call at a time, each nkilib conv call takes ~35 ms instead of the ~4 ms it takes
in the async pipeline (timed decode 22.5 s vs 2.2 s untimed). For device time per graph use the
system trace (scripts/prof/sys_phases.py) and match its NEFFs by count per decode.
Usage: python3 vae_op_profile.py   Env: TH (15), TW (21), VAE_DEEP_OUT (json of call windows),
       plus the VAE flags of the pipeline.
"""
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TH, TW = int(os.environ.get("TH", "15")), int(os.environ.get("TW", "21"))
T_LAT = 21

excl, cnt = defaultdict(float), defaultdict(int)
stack = []
ON = {"on": False}
WINDOWS = []  # (t0_ns, t1_ns, key, depth) wall-clock (time.time_ns) of every timed call
OUT = os.environ.get("VAE_DEEP_OUT", "")  # write the windows here (json) for trace attribution


def sync():
    torch.neuron.synchronize()


def timed(key_fn, fn):
    def w(*a, **k):
        if not ON["on"]:
            return fn(*a, **k)
        sync()
        key = key_fn(*a, **k)
        stack.append([key, time.perf_counter(), 0.0, time.time_ns()])
        out = fn(*a, **k)
        sync()
        key, t0, child, w0 = stack.pop()
        WINDOWS.append((w0, time.time_ns(), key, len(stack)))
        dt = time.perf_counter() - t0
        excl[key] += dt - child
        cnt[key] += 1
        if stack:
            stack[-1][2] += dt
        return out
    return w


def hw(t):
    return f"{t.shape[-2]}x{t.shape[-1]}"


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    import wan.modules.vae2_2 as V
    vae = V.Wan2_2_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.2_VAE.pth"), device=torch.device("cpu"))
    vae.model = vae.model.to(device=dev, dtype=torch.bfloat16)
    vae.scale = [s.to(device=dev, dtype=torch.bfloat16) for s in vae.scale]

    # compiled RMS_norm + SiLU graphs (module globals, looked up at call time)
    for name, label in (("_rms_silu", "norm+silu"), ("_rms_silu_cat", "norm+silu+cache-cat"),
                        ("_rms_silu_pad", "norm+silu+cache-cat")):
        f = getattr(V, name)
        if f is not None:
            setattr(V, name, timed(lambda x, *a, _l=label, **k: f"{_l} C={x.shape[1]} @{hw(x)}", f))

    def conv_key(m):
        kd, kh, kw = m.kernel_size
        def key(x, *a, **k):
            path = "nki" if getattr(m, "_nki_filters", None) is not None else "eager"
            return f"conv{kd}x{kh}x{kw} {path} {m.in_channels}->{m.out_channels} @{hw(x)}"
        return key



    for m in vae.model.modules():
        if isinstance(m, V.CausalConv3d):
            m.forward = timed(conv_key(m), m.forward)
            m.forward_time_padded = timed(conv_key(m), m.forward_time_padded)
        elif isinstance(m, V.Resample):
            m.forward = timed(lambda x, *a, _m=m, **k: f"resample-glue {_m.mode} C={x.shape[1]} @{hw(x)}", m.forward)
            m._subpixel = timed(lambda x, *a, **k: f"subpixel-upsample-conv C={x.shape[1]} @{hw(x)}->2x", m._subpixel)
        elif isinstance(m, V.DupUp3D):
            m.forward = timed(lambda x, *a, _m=m, **k: f"dupup-shortcut {x.shape[1]}->{_m.out_channels} @{hw(x)}", m.forward)
        elif isinstance(m, V.ResidualBlock):
            m.forward = timed(lambda x, *a, _m=m, **k: f"resblock-glue {_m.in_dim}->{_m.out_dim} @{hw(x)}", m.forward)
        elif isinstance(m, V.Up_ResidualBlock):
            m.forward = timed(lambda x, *a, **k: f"up-stage-glue (clone+add) @{hw(x)}", m.forward)
        elif isinstance(m, V.AttentionBlock):
            m.forward = timed(lambda x, *a, **k: f"attention-block @{hw(x)}", m.forward)
        elif isinstance(m, V.Decoder3d):
            m.forward = timed(lambda x, *a, **k: "decoder-glue", m.forward)

    g = torch.Generator().manual_seed(0)
    z = torch.randn(48, T_LAT, TH, TW, generator=g).to(torch.bfloat16).to(dev)
    marks = {}
    for rep in range(2):  # warmup/compile, then one timed decode
        ON["on"] = rep == 1
        excl.clear()
        cnt.clear()
        WINDOWS.clear()
        sync()
        t0 = time.perf_counter()
        w0 = time.time_ns()
        vae.decode([z])
        sync()
        tot = time.perf_counter() - t0
        marks["timed" if rep == 1 else "warmup"] = (w0, time.time_ns())
        print(f"[vae_deep] {'warmup' if rep == 0 else 'timed'} decode {tot:.3f}s", flush=True)
    ON["on"] = False
    time.sleep(1.0)  # > 0.5 s idle gap separates the untimed decode in the trace
    sync()
    t0 = time.perf_counter()
    w0 = time.time_ns()
    vae.decode([z])
    sync()
    marks["untimed"] = (w0, time.time_ns())
    print(f"[vae_deep] untimed decode {time.perf_counter() - t0:.3f}s", flush=True)
    if OUT:
        import json
        json.dump({"marks": marks, "windows": WINDOWS, "monotonic_minus_wall_ns":
                   time.monotonic_ns() - time.time_ns()}, open(OUT, "w"))
        print(f"[vae_deep] wrote {len(WINDOWS)} call windows to {OUT}", flush=True)
    s = sum(excl.values())
    excl["outer (decode loop: conv2, cat, unpatchify, clamp)"] = tot - s
    cat = defaultdict(float)
    for k, v in excl.items():
        cat[k.split(" ")[0]] += v
    print(f"[vae_deep] by category (timed total {tot * 1e3:.1f} ms):", flush=True)
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<28} {v * 1e3:8.1f} ms  {100 * v / tot:5.1f}%", flush=True)
    print("[vae_deep] by call site:", flush=True)
    for k, v in sorted(excl.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<60} {v * 1e3:8.1f} ms  n={cnt.get(k, 0):4d}  {100 * v / tot:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
