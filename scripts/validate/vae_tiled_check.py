"""Decode one dumped final latent with the pipeline's tiled VAE (all ranks), save the frames.

Isolates VAE changes from the DiT: every configuration decodes the SAME latent.
torchrun --nproc_per_node=8 vae_tiled_check.py <dump_with_latent.pt> <out_dir> [<ref_frames.npy> ...]
Env: VAE flags as in the pipeline (VAE_FUSE_NORM, VAE_TCHUNK, VAE_TILE_OVERLAP, ...), REPS (2).
Rank 0 writes <out_dir>/frames.npy (uint8 [F, H, W, 3]) and prints the warm decode time and the
PSNR against each reference frames.npy given.
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    src, out_dir, refs = sys.argv[1], sys.argv[2], sys.argv[3:]
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    if rank == 0:
        torch.set_num_threads(16)
    dev = torch.device("neuron")
    from models.vae_tiled import decode_tiled
    from wan.modules.vae2_2 import Wan2_2_VAE
    cpu_group = dist.new_group(backend="gloo")
    vae = Wan2_2_VAE(vae_pth=os.path.join(MODEL_PATH, "Wan2.2_VAE.pth"), device=torch.device("cpu"))
    vae.model = vae.model.to(device=dev, dtype=torch.bfloat16)
    vae.scale = [s.to(device=dev, dtype=torch.bfloat16) for s in vae.scale]
    vae.prepack_nki_filters()  # as the pipeline: rep 0 decodes like every later rep
    z = torch.load(os.path.join(src, "latent.pt")).to(torch.bfloat16).to(dev)
    overlap = int(os.environ.get("VAE_TILE_OVERLAP", "4"))
    for rep in range(int(os.environ.get("REPS", "2"))):
        dist.barrier()
        t0 = time.time()
        video = decode_tiled(vae, z, cpu_group, overlap)
        dt = time.time() - t0
        if rank == 0:
            print(f"[vae_tiled_check] rep {rep}: decode_tiled {dt:.2f}s", flush=True)
    if rank == 0:
        fr = video.add_(1).mul_(127.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "frames.npy"), fr)
        for r in refs:
            ref = np.load(r)
            per = [psnr(fr[i], ref[i]) for i in range(fr.shape[0])]
            print(f"[vae_tiled_check] vs {r}: PSNR {psnr(fr, ref):.2f} dB, worst frame {min(per):.2f} dB "
                  f"(#{int(np.argmin(per))})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
