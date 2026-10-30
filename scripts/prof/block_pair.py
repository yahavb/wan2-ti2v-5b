"""One DiT block, compiled as in production, on ONE core with no collectives: a NEFF that
`neuron-explorer capture --single-io` can run, so the DiT gets a NEFF+NTFF pair for Neuron Explorer.

The production block graph holds in-graph collectives (all-reduce / all-gather / reduce-scatter),
and a collective graph cannot be replayed alone. Here the block's weights are sharded exactly as
on one production rank, but the TP group has size 1, so models.fast_dit drops every collective:
same matmul, attention and elementwise work per core, minus the communication.
  CFG_PARALLEL=1 -> TP=4 shard, batch 1 (the production CFG-parallel layout)
  CFG_PARALLEL=0 -> TP=8 shard, batch 2 (both CFG passes batched)
Both give one core 6 (batch x head) attention instances of 7854 tokens and the same matmul FLOPs.
Self-attention: FAST_ATTN_IMPL (sdpa | cte | constmax); cross: FAST_CROSS_IMPL. Uses the
non-SP block (_FastBlock); FAST_SP only moves the elementwise work onto a token shard.
The QK-norm is local (no global RMS), so the numbers are a timing probe, not the model output.

NEURON_RT_VISIBLE_CORES=0 torchrun --nproc_per_node=1 --master_port=29501 scripts/prof/block_pair.py
Env: LAYER (0), REPS (10)
"""
import os
import sys
import time

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "validate"))
from block_bench import GRID, L, DIM, D, LC, load_block, rope_freqs  # noqa: E402


def main():
    dist.init_process_group(backend="neuron")
    torch.neuron.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    import models.fast_dit as fd
    from models.tp_utils import init_tp_group, shard_model_tp

    cfg_par = os.environ.get("CFG_PARALLEL", "0") == "1"
    shard_tp, bsz = (4, 1) if cfg_par else (8, 2)
    impl = fd.FAST_ATTN_IMPL
    layer = int(os.environ.get("LAYER", "0"))
    reps = int(os.environ.get("REPS", "10"))

    blk, _ = load_block(layer)

    class _M:  # minimal model stand-in for shard_model_tp
        pass
    m = _M()
    m.blocks, m.num_heads, m.parameters = [blk], 24, blk.parameters
    init_tp_group(tp_degree=1)          # group of one: fast_dit skips every collective
    shard_model_tp(m, 0, shard_tp)      # ...but the weights are one production rank's shard
    blk = blk.to(torch.bfloat16).to(dev)

    g = torch.Generator().manual_seed(0)
    x = (torch.randn(bsz, L, DIM, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    e0 = (torch.randn(1, GRID[0], 6, DIM, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    freqs = torch.cat([rope_freqs(1024, D - 4 * (D // 6)), rope_freqs(1024, 2 * (D // 6)),
                       rope_freqs(1024, 2 * (D // 6))], 1)
    cos, sin = fd._rope_tables(freqs, GRID, D, dev)
    ctx = (torch.randn(bsz, LC, DIM, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    ck, cv = torch.compile(fd._FastCrossKV(blk), **kw)(ctx)
    block = torch.compile(fd._FastBlock(blk, GRID[0], impl, None), **kw)

    for _ in range(3):  # compile + warm-up
        y = block(x, e0, cos, sin, ck, cv)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        y = block(x, e0, cos, sin, ck, cv)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1e3
    fd.FastWanDiT.log_graph_breaks()
    print(f"[block_pair] layer {layer} shard TP={shard_tp} batch={bsz} self={impl} "
          f"cross={fd.FAST_CROSS_IMPL} heads/core={24 // shard_tp} L={L}: {ms:.2f} ms per block "
          f"(no collectives) out {tuple(y.shape)} finite={bool(torch.isfinite(y.float()).all())}",
          flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
