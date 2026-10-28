"""x264 encode time / size / PSNR of a dumped video at several presets (CPU only).
python3 mp4_bench.py <frames.npy> [preset:crf ...]   default: veryfast:20 superfast:20 ultrafast:20 ultrafast:17"""
import math
import os
import sys
import time

import imageio
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    fr = np.load(sys.argv[1])
    cfgs = sys.argv[2:] or ["veryfast:20", "superfast:20", "ultrafast:20", "ultrafast:17"]
    for cfg in cfgs:
        preset, crf = cfg.split(":")
        out = f"/tmp/mp4_bench_{preset}_{crf}.mp4"
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            imageio.mimwrite(out, list(fr), fps=24, codec="libx264", quality=None,
                             ffmpeg_params=["-preset", preset, "-crf", crf])
            ts.append(time.perf_counter() - t0)
        dec = np.stack(imageio.mimread(out, memtest=False))
        print(f"{preset:>10} crf {crf}: write {min(ts):.3f}s (min of 3)  size {os.path.getsize(out) / 1e6:.2f} MB  "
              f"PSNR vs raw {psnr(dec[:fr.shape[0]], fr):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
