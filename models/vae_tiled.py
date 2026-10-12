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
import time

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


def decode_tiled(vae, z, cpu_group, overlap=8):
    """vae: Wan2_2_VAE on this rank's device; z: [C, T, H, W] latent (same on all ranks).

    Returns the decoded video [3, F, H*s, W*s] (float32, clamped) on rank 0, None elsewhere.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    _, _, h, w = z.shape
    th, tw, ys, xs = tile_grid(h, w, world, overlap)
    tiles = [(i, j) for i in range(len(ys)) for j in range(len(xs))]

    t0 = time.time()
    mine = []
    for k, (i, j) in enumerate(tiles):
        if k % world != rank:
            continue
        zt = z[:, :, ys[i]:ys[i] + th, xs[j]:xs[j] + tw].contiguous()
        out = vae.decode([zt])[0]  # [3, F, th*s, tw*s], float32 clamped
        mine.append((k, out.to(torch.bfloat16).cpu()))

    t_dec = time.time() - t0
    gathered = [None] * world if rank == 0 else None
    dist.gather_object((t_dec, mine), gathered, dst=0, group=cpu_group)
    if rank != 0:
        return None
    t_gather = time.time() - t0 - t_dec
    dec_times = [g[0] for g in gathered]
    gathered = [g[1] for g in gathered]
    t1 = time.time()

    parts = dict(item for lst in gathered for item in lst)
    sample = parts[0]
    s = sample.shape[-1] // tw  # spatial scale latent -> pixels
    C, F = sample.shape[0], sample.shape[1]
    H, W = h * s, w * s
    ramp = overlap * s
    acc = torch.zeros(C, F, H, W)
    wsum = torch.zeros(1, 1, H, W)
    for k, (i, j) in enumerate(tiles):
        y0, x0 = ys[i] * s, xs[j] * s
        wy = _ramp(th * s, ramp, i > 0, i < len(ys) - 1)
        wx = _ramp(tw * s, ramp, j > 0, j < len(xs) - 1)
        wt = (wy[:, None] * wx[None, :])[None, None]
        acc[:, :, y0:y0 + th * s, x0:x0 + tw * s] += parts[k].float() * wt
        wsum[:, :, y0:y0 + th * s, x0:x0 + tw * s] += wt
    out = (acc / wsum).clamp_(-1, 1)
    print(f"[vae_tiled] tiles={len(tiles)} ({th}x{tw} latent) decode max/min over ranks="
          f"{max(dec_times):.2f}/{min(dec_times):.2f}s gather(after rank0 decode)={t_gather:.2f}s "
          f"blend={time.time() - t1:.2f}s", flush=True)
    return out


__all__ = ["decode_tiled", "tile_grid"]
