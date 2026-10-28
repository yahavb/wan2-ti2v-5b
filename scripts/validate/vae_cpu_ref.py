"""Exact (untiled, fp32, CPU) Wan2.2 VAE decode of a dumped final latent: the VAE reference.

python3 vae_cpu_ref.py <dump_dir_with_latent.pt> <out_dir> [<frames.npy to score> ...]
Writes <out_dir>/frames.npy (uint8 [F, H, W, 3], same conversion as the pipeline) and prints the
PSNR of every given frames.npy against it. Run with USE_NKI_VAE=0 USE_NKI_CONV3D=0 and the VAE_* fusion flags at 0 (they torch.compile for Neuron).
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    src, out_dir, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    out_path = os.path.join(out_dir, "frames.npy")
    if not os.path.exists(out_path):
        torch.set_num_threads(int(os.environ.get("THREADS", "64")))
        torch.set_grad_enabled(False)
        from wan.modules.vae2_2 import Wan2_2_VAE
        vae = Wan2_2_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.2_VAE.pth"), device=torch.device("cpu"))
        vae.model = vae.model.float()
        z = torch.load(os.path.join(src, "latent.pt")).float()
        t0 = time.time()
        out = vae.decode([z])[0]  # float32 [3, F, H, W] in [-1, 1]
        print(f"[vae_cpu_ref] fp32 CPU decode {time.time() - t0:.0f}s {tuple(out.shape)}", flush=True)
        fr = out.add_(1).mul_(127.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()
        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, fr)
    ref = np.load(out_path)
    for p in others:
        fr = np.load(p)
        per = [psnr(fr[i], ref[i]) for i in range(ref.shape[0])]
        print(f"[vae_cpu_ref] {p}: PSNR {psnr(fr, ref):.2f} dB, worst frame {min(per):.2f} dB "
              f"(#{int(np.argmin(per))}), frame 40 {per[40]:.2f} dB", flush=True)


if __name__ == "__main__":
    main()
