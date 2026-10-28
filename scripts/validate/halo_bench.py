"""Cost of an exact 1-px halo exchange fused into the VAE's compiled norm graph (8 ranks, 4x2 tiles).
torchrun --nproc_per_node=8 halo_bench.py
Per stage shape (C, T new frames, core tile h x w):
  norm_cat       today's VAE_FUSE_CAT graph: z = cat(prev, rms_silu(x)) on the core tile
  halo_norm_cat  same graph plus an in-graph all_gather of every rank's 4 edges + 4 corners and a
                 one-hot pick of the 8 neighbours -> z on the (h+2) x (w+2) haloed tile
The one-hot vectors are inputs (all-zero for a missing neighbour = zero padding), so every rank
runs the same graph. Correctness: the haloed tile is checked against the full frame.
MODE=strips: 8x1 strips of the frame instead; only the row above and the row below are exchanged
(strip_halo_norm_cat). Env: MODE (tiles), REPS (20).
"""
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

REPS = int(os.environ.get("REPS", "20"))
ROWS, COLS = 4, 2
SHAPES = [  # (name, C, T, h, w): core tile of the 44x34 latent frame at each decoder resolution
    ("s8_c256", 256, 4, 88, 136),
    ("s4_c512", 512, 4, 44, 68),
    ("s2_c1024", 1024, 2, 22, 34),
    ("s1_c1024", 1024, 1, 11, 17),
]


def rms_silu(x, gamma):
    rms = x.float().pow(2).mean(1, keepdim=True).clamp(min=1e-8).rsqrt()
    return torch.nn.functional.silu((x * rms).to(gamma.dtype) * gamma)


def norm_cat(x, gamma, prev):
    z = torch.cat([prev, rms_silu(x, gamma)], dim=2)
    return z, z[:, :, -2:].clone()


def make_halo_norm_cat(group):
    def halo_norm_cat(x, gamma, prev, oh):
        # x [1, C, T, h, w]; oh [8 slots, world] one-hot of the neighbour feeding each slot
        # slots: 0 top, 1 bottom, 2 left, 3 right, 4 top-left, 5 top-right, 6 bottom-left, 7 bottom-right
        _, C, T, h, w = x.shape
        edges = torch.cat([x[..., 0, :], x[..., h - 1, :],                   # [1, C, T, w] x2 rows
                           x[..., :, 0], x[..., :, w - 1],                   # [1, C, T, h] x2 cols
                           x[..., 0:1, 0], x[..., 0:1, w - 1],               # corners [1, C, T, 1]
                           x[..., h - 1:h, 0], x[..., h - 1:h, w - 1]], dim=-1)  # [1, C, T, 2w+2h+4]
        g = funcol.all_gather_tensor(edges, 0, group)                         # [world, C, T, E]
        pick = torch.einsum("sr,rcte->scte", oh.to(g.dtype), g)              # [8 slots, C, T, E]
        o_b, o_l, o_r = w, 2 * w, 2 * w + h
        o_c = 2 * w + 2 * h
        top = pick[0, :, :, o_b:o_b + w]          # bottom row of the tile above
        bot = pick[1, :, :, 0:w]                  # top row of the tile below
        left = pick[2, :, :, o_r:o_r + h]         # right column of the tile on the left
        right = pick[3, :, :, o_l:o_l + h]        # left column of the tile on the right
        tl = pick[4, :, :, o_c + 3:o_c + 4]       # bottom-right corner of the top-left tile
        tr = pick[5, :, :, o_c + 2:o_c + 3]
        bl = pick[6, :, :, o_c + 1:o_c + 2]
        br = pick[7, :, :, o_c + 0:o_c + 1]
        mid = torch.cat([left.unsqueeze(-1), x[0], right.unsqueeze(-1)], dim=-1)  # [C, T, h, w+2]
        row_t = torch.cat([tl, top, tr], dim=-1).unsqueeze(2)                       # [C, T, 1, w+2]
        row_b = torch.cat([bl, bot, br], dim=-1).unsqueeze(2)
        xh = torch.cat([row_t, mid, row_b], dim=2).unsqueeze(0)                     # [1, C, T, h+2, w+2]
        z = torch.cat([prev, rms_silu(xh, gamma)], dim=2)
        return z, z[:, :, -2:].clone()
    return halo_norm_cat


def make_strip_halo_norm_cat(group):
    """8x1 strips: only the row above and the row below are exchanged (contiguous rows); the
    nkilib conv pads W itself."""
    def strip_halo_norm_cat(x, gamma, prev, oh):  # oh [2 slots (above, below), world]
        _, C, T, h, w = x.shape
        edges = torch.cat([x[..., 0:1, :], x[..., h - 1:h, :]], dim=-2)     # [1, C, T, 2, w]
        g = funcol.all_gather_tensor(edges, 0, group)                         # [world, C, T, 2, w]
        pick = torch.einsum("sr,rctkw->sctkw", oh.to(g.dtype), g)            # [2, C, T, 2, w]
        xh = torch.cat([pick[0, :, :, 1:2], x[0], pick[1, :, :, 0:1]], dim=2).unsqueeze(0)
        z = torch.cat([prev, rms_silu(xh, gamma)], dim=2)                     # [1, C, 2+T, h+2, w]
        return z, z[:, :, -2:].clone()
    return strip_halo_norm_cat


STRIPS = [  # (name, C, T, h, w): 8 strips of the 44x34 latent frame (s1 rounded up to 6 rows)
    ("s8_c256", 256, 4, 44, 272),
    ("s4_c512", 512, 4, 22, 136),
    ("s2_c1024", 1024, 2, 11, 68),
    ("s1_c1024", 1024, 1, 6, 34),
]


def strips_main(rank, world, dev, group):
    f_nc = torch.compile(norm_cat, backend="neuron", dynamic=False)
    f_s = torch.compile(make_strip_halo_norm_cat(group), backend="neuron", dynamic=False)
    oh = torch.zeros(2, world)
    if rank > 0:
        oh[0, rank - 1] = 1.0
    if rank < world - 1:
        oh[1, rank + 1] = 1.0
    oh = oh.to(dev)
    for name, C, T, h, w in STRIPS:
        g = torch.Generator().manual_seed(0)
        full = torch.randn(1, C, T, world * h, w, generator=g).to(torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=g)).to(torch.bfloat16)
        x = full[..., rank * h:(rank + 1) * h, :].contiguous().to(dev)
        prev = torch.randn(1, C, 2, h, w, generator=g).to(torch.bfloat16).to(dev)
        prev_h = torch.randn(1, C, 2, h + 2, w, generator=g).to(torch.bfloat16).to(dev)
        gd = gamma.to(dev)
        ms_nc, _ = timeit(f_nc, x, gd, prev)
        ms_s, (z, _) = timeit(f_s, x, gd, prev_h, oh)
        fp = torch.nn.functional.pad(full, (0, 0, 1, 1))
        ref = rms_silu(fp[..., rank * h:rank * h + h + 2, :].float(), gamma.float())
        err = (z[:, :, 2:].float().cpu() - ref).abs().max().item()
        errs = [None] * world
        dist.all_gather_object(errs, err)
        if rank == 0:
            print(f"strip {name:>9} C={C} T={T} strip {h}x{w}: norm_cat {ms_nc:.3f} ms, strip_halo_norm_cat "
                  f"{ms_s:.3f} ms (+{ms_s - ms_nc:.3f}); max|err| vs full frame over ranks {max(errs):.4f}",
                  flush=True)


def neighbours(rank):
    r, c = divmod(rank, COLS)
    oh = torch.zeros(8, ROWS * COLS)
    for s, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]):
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            oh[s, rr * COLS + cc] = 1.0
    return oh


def timeit(f, *a):
    f(*a)
    torch.neuron.synchronize()
    for _ in range(3):
        f(*a)
    torch.neuron.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == ROWS * COLS
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("neuron")
    group = dist.group.WORLD
    if os.environ.get("MODE", "tiles") == "strips":
        strips_main(rank, world, dev, group)
        dist.destroy_process_group()
        return
    r, c = divmod(rank, COLS)
    f_nc = torch.compile(norm_cat, backend="neuron", dynamic=False)
    f_h = torch.compile(make_halo_norm_cat(group), backend="neuron", dynamic=False)
    for name, C, T, h, w in SHAPES:
        g = torch.Generator().manual_seed(0)
        full = torch.randn(1, C, T, ROWS * h, COLS * w, generator=g).to(torch.bfloat16)  # same on all ranks
        gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=g)).to(torch.bfloat16)
        x = full[..., r * h:(r + 1) * h, c * w:(c + 1) * w].contiguous().to(dev)
        prev = torch.randn(1, C, 2, h, w, generator=g).to(torch.bfloat16).to(dev)
        prev_h = torch.randn(1, C, 2, h + 2, w + 2, generator=g).to(torch.bfloat16).to(dev)
        gd, oh = gamma.to(dev), neighbours(rank).to(dev)
        ms_nc, _ = timeit(f_nc, x, gd, prev)
        ms_h, (z, _) = timeit(f_h, x, gd, prev_h, oh)
        # reference haloed tile from the full frame (zero padded outside)
        fp = torch.nn.functional.pad(full, (1, 1, 1, 1))
        ref = rms_silu(fp[..., r * h:r * h + h + 2, c * w:c * w + w + 2].float(), gamma.float())
        err = (z[:, :, 2:].float().cpu() - ref).abs().max().item()
        if rank == 0:
            print(f"{name:>9} C={C} T={T} tile {h}x{w}: norm_cat {ms_nc:.3f} ms, halo_norm_cat {ms_h:.3f} ms "
                  f"(+{ms_h - ms_nc:.3f}); haloed tile max|err| vs full frame {err:.4f}", flush=True)
        errs = [None] * world
        dist.all_gather_object(errs, err)
        if rank == 0:
            print(f"           max|err| over ranks {max(errs):.4f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
