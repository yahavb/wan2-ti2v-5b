"""Compare DUMP_BLOCKS dumps (models/fast_dit.py) of pipeline runs: per block, the rel-RMS between
runs, and for block 0 (cond group) the error of each run against that block in fp32 on the CPU,
computed from the run's own input with ref_fp32.block_fwd (exact attention over the 512 keys).
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 block_dump_cmp.py <step0 dump dir> <blocks dir A> <blocks dir B> ...
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")


def full(d, i, group, key):
    """Concatenate the shards of CFG group `group` for block i -> [N_pad, C] (shard order)."""
    parts = {}
    for f in glob.glob(os.path.join(d, f"blk{i}_rank*.pt")):
        r = torch.load(f)
        if r["group"] == group:
            parts[r["shard"]] = r[key]
    return torch.cat([parts[s] for s in sorted(parts)], 0).float() if parts else None


def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def main():
    step0, runs = sys.argv[1], sys.argv[2:]
    blocks = sorted({int(os.path.basename(f)[3:].split("_")[0]) for f in glob.glob(os.path.join(runs[0], "blk*_rank*.pt"))})
    names = [os.path.basename(r.rstrip("/")) for r in runs]
    for i in blocks:
        for g in (0, 1):
            outs = [full(r, i, g, "x_out") for r in runs]
            line = f"block {i:2d} group {g}: |x_out| rms {outs[0].pow(2).mean().sqrt().item():.3f}"
            for n, o in zip(names[1:], outs[1:]):
                line += f"  {n} vs {names[0]} {rel(o, outs[0]):.5f}"
            print(line, flush=True)
    # block 0, cond group: fp32 truth from each run's own block input
    torch.set_num_threads(int(os.environ.get("THREADS", "32")))
    torch.set_grad_enabled(False)
    import models.fast_dit as fd
    from ref_fp32 import block_fwd, lin, rms
    from wan.modules.model import WanModel
    s0 = torch.load(os.path.join(step0, "step0.pt"))
    model = WanModel.from_pretrained(MODEL_PATH).eval().float()
    lat = s0["latent_in"]
    grid = (lat.shape[1], lat.shape[2] // 2, lat.shape[3] // 2)
    L = grid[0] * grid[1] * grid[2]
    cos, sin = fd._rope_tables(model.freqs, grid, model.dim // model.num_heads, torch.device("cpu"))
    blk = model.blocks[blocks[0]]
    ca = blk.cross_attn
    kv = []
    for c in (s0["ctx"], s0["ctx_null"]):  # group 0 = cond, group 1 = uncond
        context = model.text_embedding(c.float().unsqueeze(0))
        kv.append((rms(lin(context, ca.k), ca.norm_k).view(1, -1, ca.num_heads, ca.head_dim),
                   lin(context, ca.v).view(1, -1, ca.num_heads, ca.head_dim)))
    for n, r in zip(names, runs):
        e0 = torch.load(glob.glob(os.path.join(r, f"blk{blocks[0]}_rank*.pt"))[0])["e0"].float()
        errs = []
        for g in (0, 1):
            x_in = full(r, blocks[0], g, "x_in")[:L].unsqueeze(0)
            x_out = full(r, blocks[0], g, "x_out")[:L]
            y32 = block_fwd(blk, x_in, e0, cos, sin, kv[g][0], kv[g][1], grid[0])[0]
            errs.append(x_out - y32)
            print(f"{n}: block {blocks[0]} group {g} vs fp32 on its own input: {rel(x_out, y32):.5f}; "
                  f"update vs fp32 update: {rel(x_out - x_in[0], y32 - x_in[0]):.5f}", flush=True)
        a, b = errs[0].flatten(), errs[1].flatten()
        rho = (a @ b / (a.norm() * b.norm())).item()
        cfg = (5 * errs[0] - 4 * errs[1]).pow(2).mean().sqrt() / errs[0].pow(2).mean().sqrt()
        print(f"{n}: cond / uncond error correlation {rho:.3f}; |5 e_c - 4 e_u| / |e_c| = {cfg.item():.2f}", flush=True)


if __name__ == "__main__":
    main()
