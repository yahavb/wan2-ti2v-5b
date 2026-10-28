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


def _wait_for(path, deadline):
    while not os.path.exists(path):
        if time.time() > deadline:
            raise TimeoutError(f"tile chunk {path} never arrived")
        time.sleep(0.001)


def _assemble(tag, tiles, ys, xs, th, tw, s, h, w, n_chunks, writer, finish, done):
    """Rank 0 helper thread: for each latent-frame chunk, sum the 8 weighted tile pieces into
    place (same order and math as decode_tiled), convert to uint8 frames via finish(), and feed
    them to the mp4 writer. done = dict that receives frames / timing / the first exception."""
    try:
        deadline = time.time() + 600
        frames = []
        for c in range(n_chunks):
            acc = None
            for k, (i, j) in enumerate(tiles):
                path = f"{tag}_{k}_{c}.bin"
                _wait_for(path, deadline)
                part = torch.from_numpy(np.fromfile(path, dtype=np.int16)).view(torch.bfloat16)
                part = part.view(3, -1, th * s, tw * s)
                if acc is None:
                    acc = torch.zeros(3, part.shape[1], h * s, w * s)
                acc[:, :, ys[i] * s:ys[i] * s + th * s, xs[j] * s:xs[j] * s + tw * s] += part.float()
                os.remove(path)
            if c == n_chunks - 1:
                done["t_last_chunk"] = time.time()
            v = finish(acc.clamp_(-1, 1))  # [f, H, W, 3] uint8 numpy
            for fr in v:
                writer.append_data(fr)
            frames.append(v)
        t0 = time.time()
        writer.close()
        done["t_close"] = time.time() - t0
        done["frames"] = np.concatenate(frames, 0)
    except BaseException as ex:  # noqa: BLE001 - re-raised by the caller
        done["error"] = ex


def decode_tiled_stream(vae, z, cpu_group, overlap, make_writer, finish):
    """decode_tiled() that streams: every rank ships each decoded latent-frame chunk of its tile
    (weighted, bf16) to shared memory right away, and a rank-0 thread blends chunk by chunk and
    feeds the frames to the mp4 encoder while the tiles are still decoding. The blend is the same
    per-element math as decode_tiled, so the frames are identical.
    make_writer(): returns an imageio-style writer (append_data / close), called on rank 0.
    finish(acc): blended [3, f, H, W] float32 chunk -> uint8 numpy [f, H, W, 3].
    Returns the uint8 video [F, H, W, 3] on rank 0 (already written by the writer), None elsewhere.
    """
    import threading
    rank, world = dist.get_rank(), dist.get_world_size()
    _, n_chunks, h, w = z.shape
    th, tw, ys, xs = tile_grid(h, w, world, overlap)
    tiles = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    _CALL[0] += 1
    tag = f"{SHM_DIR}/wan_vae_stream_{os.getppid()}_{_CALL[0]}"
    done, thread = {}, None
    if rank == 0:
        thread = threading.Thread(
            target=_assemble, args=(tag, tiles, ys, xs, th, tw, 16, h, w, n_chunks, make_writer(),
                                    finish, done), daemon=True)
        thread.start()
    t0 = time.time()

    def ship(k, c, part):
        p = f"{tag}_{k}_{c}.bin"
        part.cpu().view(torch.int16).numpy().tofile(p + ".tmp")
        os.rename(p + ".tmp", p)  # atomic: the reader never sees a partial file

    for k, (i, j) in enumerate(tiles):
        if k % world != rank:
            continue
        zt = z[:, :, ys[i]:ys[i] + th, xs[j]:xs[j] + tw].contiguous()
        wt, pending = None, None
        for c, out in enumerate(vae.decode_chunks(zt)):  # [3, f, th*s, tw*s] on device
            if wt is None:
                s = out.shape[-1] // tw
                assert s == 16, s
                wt = _weights(tiles, ys, xs, th, tw, s, overlap * s, h * s, w * s)[k]
                wt = wt.to(out.device, out.dtype)
            part = (out * wt).to(torch.bfloat16)
            if pending is not None:  # copy the previous chunk while this one is queued
                ship(k, *pending)
            pending = (c, part)
        ship(k, *pending)
    t_dec = time.time() - t0
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(t_dec, gathered, dst=0, group=cpu_group)
    if rank != 0:
        return None
    t1 = time.time()
    thread.join()
    if "error" in done:
        raise done["error"]
    print(f"[vae_tiled] stream tiles={len(tiles)} ({th}x{tw} latent, overlap {overlap}) decode "
          f"max/min over ranks={max(gathered):.2f}/{min(gathered):.2f}s; tail after decode "
          f"{time.time() - t1:.2f}s (writer close {done['t_close']:.2f}s)", flush=True)
    return done["frames"]


__all__ = ["decode_tiled", "decode_tiled_stream", "tile_grid"]
