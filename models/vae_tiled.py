"""Spatially tiled Wan2.2 VAE decode spread across ranks (eager VAE on every rank).

Each rank decodes whole-clip (all frames, so the causal temporal path is exact) for a
subset of overlapping spatial tiles of the latent; tiles are gathered to rank 0 over a
CPU (gloo) group and blended with linear ramps in the overlaps. Same idea as
vllm-omni-neuron's patch-parallel decode / diffusers tiled_decode; like those, the
result is close to but not bit-identical with a full-frame decode near tile seams.

All tiles have the same latent size (boundary windows are shifted inward), so every
rank runs the same op shapes.
"""

import math
import os
import time

import numpy as np

import torch
import torch.distributed as dist


def tile_grid(h, w, world, overlap):
    """Pick a rows x cols grid (rows*cols <= world) minimizing tile area.

    Returns (th, tw, ys, xs): tile size and top-left positions in latent units.
    """
    best = None
    for r in range(1, world + 1):
        for c in range(1, world // r + 1):
            th = min(h, math.ceil(h / r) + (overlap if r > 1 else 0))
            tw = min(w, math.ceil(w / c) + (overlap if c > 1 else 0))
            key = (th * tw, abs(th - tw), -(r * c))
            if best is None or key < best[0]:
                best = (key, r, c, th, tw)
    _, r, c, th, tw = best
    ys = [0] if r == 1 else [round(k * (h - th) / (r - 1)) for k in range(r)]
    xs = [0] if c == 1 else [round(k * (w - tw) / (c - 1)) for k in range(c)]
    return th, tw, ys, xs


def _ramp(n, ramp, lo_edge, hi_edge):
    """1-D blend weights of length n: linear ramps on sides that border another tile."""
    # min() of the two ramps stays continuous even when they overlap (tile < 2*ramp).
    pos = torch.arange(n, dtype=torch.float32).add_(0.5)
    wgt = torch.ones(n)
    if ramp > 0 and lo_edge:
        wgt = torch.minimum(wgt, pos / ramp)
    if ramp > 0 and hi_edge:
        wgt = torch.minimum(wgt, pos.flip(0) / ramp)
    return wgt


_CALL = [0]
SHM_DIR = os.environ.get("VAE_TILE_SHM_DIR", "/dev/shm")


def _weights(tiles, ys, xs, th, tw, s, ramp, H, W):
    """Per-tile blend weights pre-normalized by their sum, so tiles only need adding."""
    wts, wsum = [], torch.zeros(H, W)
    for i, j in tiles:
        wy = _ramp(th * s, ramp, i > 0, i < len(ys) - 1)
        wx = _ramp(tw * s, ramp, j > 0, j < len(xs) - 1)
        wt = wy[:, None] * wx[None, :]
        wsum[ys[i] * s:ys[i] * s + th * s, xs[j] * s:xs[j] * s + tw * s] += wt
        wts.append(wt)
    return [wt / wsum[ys[i] * s:ys[i] * s + th * s, xs[j] * s:xs[j] * s + tw * s]
            for wt, (i, j) in zip(wts, tiles)]


def decode_tiled(vae, z, cpu_group, overlap=8):
    """vae: Wan2_2_VAE on this rank's device; z: [C, T, H, W] latent (same on all ranks).

    Each rank decodes its tiles, multiplies by the normalized blend weight on device, and
    writes the result to shared memory; rank 0 sums the tiles into place.
    Returns the decoded video [3, F, H*s, W*s] (float32, clamped) on rank 0, None elsewhere.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    _, _, h, w = z.shape
    th, tw, ys, xs = tile_grid(h, w, world, overlap)
    tiles = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    _CALL[0] += 1
    tag = f"{SHM_DIR}/wan_vae_tile_{os.getppid()}_{_CALL[0]}"

    t0 = time.time()
    meta = None
    for k, (i, j) in enumerate(tiles):
        if k % world != rank:
            continue
        zt = z[:, :, ys[i]:ys[i] + th, xs[j]:xs[j] + tw].contiguous()
        out = vae.decode([zt])[0]  # [3, F, th*s, tw*s], float32 clamped, on device
        C, F = out.shape[0], out.shape[1]
        s = out.shape[-1] // tw
        wt = _weights(tiles, ys, xs, th, tw, s, overlap * s, h * s, w * s)[k]
        out = (out * wt.to(out.device, out.dtype)).to(torch.bfloat16).cpu()
        out.view(torch.int16).numpy().tofile(f"{tag}_{k}.bin")
        meta = (C, F, s)
    t_dec = time.time() - t0

    gathered = [None] * world if rank == 0 else None
    dist.gather_object((t_dec, meta), gathered, dst=0, group=cpu_group)  # tiny; also a barrier
    if rank != 0:
        return None
    t1 = time.time()
    dec_times = [g[0] for g in gathered]
    C, F, s = gathered[0][1]
    acc = torch.zeros(C, F, h * s, w * s)
    for k, (i, j) in enumerate(tiles):
        f = f"{tag}_{k}.bin"
        part = torch.from_numpy(np.fromfile(f, dtype=np.int16)).view(torch.bfloat16)
        part = part.view(C, F, th * s, tw * s)
        acc[:, :, ys[i] * s:ys[i] * s + th * s, xs[j] * s:xs[j] * s + tw * s] += part.float()
        os.remove(f)
    out = acc.clamp_(-1, 1)
    print(f"[vae_tiled] tiles={len(tiles)} ({th}x{tw} latent, overlap {overlap}) decode max/min over ranks="
          f"{max(dec_times):.2f}/{min(dec_times):.2f}s collect+sum={time.time() - t1:.2f}s", flush=True)
    return out


__all__ = ["decode_tiled", "tile_grid"]
