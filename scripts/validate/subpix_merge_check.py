"""kernels/vae_subpixel_merge.py vs today's path (eager stack + permute of the 4 sub-pixel phases,
then the compiled DupUp3D shortcut + add, VAE_FUSE_SHORTCUT) at the strip shapes of the 704x544
decode with VAE_TCHUNK=4, on one NeuronCore: bit-exactness and time.
NEURON_RT_VISIBLE_CORES=<core> python3 scripts/validate/subpix_merge_check.py   Env: REPS (20)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from kernels.vae_subpixel_merge import merge_call  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
CASES = [  # name, C out, Cx (stage input), T0 (input frames), h, w, factor_t, repeats, first_chunk
    ("stage2 4x->8x", 512, 1024, 8, 22, 136, 2, 4, False),
    ("stage2 first", 512, 1024, 1, 22, 136, 2, 4, True),
    ("stage1 2x->4x", 1024, 1024, 4, 11, 68, 2, 8, False),
    ("stage1 first", 1024, 1024, 1, 11, 68, 2, 8, True),
    ("stage0 1x->2x", 1024, 1024, 4, 6, 34, 1, 4, False),
]


def dupup_add(main, x, out_channels: int, factor_t: int, factor_s: int, repeats: int, first_chunk: bool):
    B, _, T, H, W = x.shape
    s = x.repeat_interleave(repeats, dim=1)
    s = s.view(B, out_channels, factor_t, factor_s, factor_s, T, H, W)
    s = s.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, out_channels, T * factor_t, H * factor_s, W * factor_s)
    if first_chunk:
        s = s[:, :, factor_t - 1:, :, :]
    return main + s


def interleave(ys):
    B, C, T, h, w = ys[0].shape
    y = torch.stack(ys, dim=2).view(B, C, 2, 2, T, h, w)
    return y.permute(0, 1, 4, 5, 2, 6, 3).reshape(B, C, T, 2 * h, 2 * w)


def timeit(f):
    f()
    torch.neuron.synchronize()
    for _ in range(3):
        f()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    torch.neuron.set_device(0)
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    f_add = torch.compile(dupup_add, backend="neuron", dynamic=False)
    for name, C, Cx, T0, h, w, ft, rep, first in CASES:
        T2 = T0 * ft - (ft - 1 if first else 0)
        g = torch.Generator().manual_seed(0)
        ys = [torch.randn(1, C, T2, h, w, generator=g).to(torch.bfloat16).to(dev) for _ in range(4)]
        x = torch.randn(1, Cx, T0, h, w, generator=g).to(torch.bfloat16).to(dev)
        t_old, ref = timeit(lambda: f_add(interleave(ys), x, C, ft, 2, rep, first))
        t_new, out = timeit(lambda: merge_call(ys, x, ft, rep, first))
        r, o = ref.float().cpu(), out.float().cpu()
        print(f"{name:>14} C={C} T2={T2} {h}x{w} -> {2 * h}x{2 * w}: today {t_old:.3f} ms, kernel {t_new:.3f} ms; "
              f"bit-identical {torch.equal(r, o)} (max |d| {(r - o).abs().max().item():.3g})", flush=True)


if __name__ == "__main__":
    main()
