"""Compare dumped outputs (DUMP_DIR of inference_neuron_ti2v.py) against a reference dump.

Usage: python3 compare.py <ref_dir> <dir> [<dir> ...]
Prints, per dir: latent rel-RMS vs ref, video PSNR (all frames), worst-frame PSNR, and the
PSNR of frame 0 (the conditioning image frame).
"""
import math
import os
import sys

import numpy as np
import torch


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ref_dir, dirs = sys.argv[1], sys.argv[2:]
    ref_lat = torch.load(os.path.join(ref_dir, "latent.pt"))
    ref_fr = np.load(os.path.join(ref_dir, "frames.npy"))
    print(f"ref {ref_dir}: latent {tuple(ref_lat.shape)} frames {ref_fr.shape}")
    ref_s0 = os.path.join(ref_dir, "step0.pt")
    ref_s0 = torch.load(ref_s0) if os.path.exists(ref_s0) else None
    for d in dirs:
        lat = torch.load(os.path.join(d, "latent.pt"))
        fr = np.load(os.path.join(d, "frames.npy"))
        rel = ((lat - ref_lat).pow(2).mean().sqrt() / ref_lat.pow(2).mean().sqrt()).item()
        per = [psnr(fr[i], ref_fr[i]) for i in range(fr.shape[0])]
        s0 = ""
        p0 = os.path.join(d, "step0.pt")
        if ref_s0 is not None and os.path.exists(p0):
            a = torch.load(p0)
            same_in = torch.equal(a["latent_in"], ref_s0["latent_in"])
            r = ref_s0["noise_pred"]
            e = ((a["noise_pred"] - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
            s0 = f" | step0 noise_pred rel_rms {e:.5f} (same input: {same_in})"
        print(f"{os.path.basename(d.rstrip('/')):>28}: latent rel_rms {rel:.4f} | video PSNR "
              f"{psnr(fr, ref_fr):6.2f} dB, worst frame {min(per):6.2f} dB (#{int(np.argmin(per))}), "
              f"frame0 {per[0]:6.2f} dB, last {per[-1]:6.2f} dB{s0}")


if __name__ == "__main__":
    main()
