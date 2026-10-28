"""In-graph collective costs at the SP block's shapes, per CFG group of 4 (interleaved by default).
torchrun --nproc_per_node=8 sp_cc_bench.py [op ...]
  ag      all-gather token shard [1964, 3072] -> [7856, 3072]      (Megatron-SP, before QKV / FFN1)
  rs      reduce-scatter [7856, 3072] -> [1964, 3072]              (after O-proj / FFN2)
  a2a     all_to_all_single [4*1964, 768] (Ulysses head<->sequence swap of one of q/k/v/o)
  a2a4    four a2a of [4*1964, 768] in one graph (q, k, v and the output swap of one block)
  ag2     all-gather over rank PAIRS inside the group: [1964, 3072] -> [3928, 3072]
  rs2     reduce-scatter over pairs: [3928, 3072] -> [1964, 3072]
  mm      [7856, 3072] @ [3072, 3584] (an FFN1 shard), no collective
  ag||mm  ag on one input and an independent mm in the same graph
Env: GROUPS=interleave|contiguous, REPS.
"""
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
S, D, N = 1964, 3072, 3584


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("neuron")
    from models.tp_utils import init_tp_group, get_tp_group
    inter = os.environ.get("GROUPS", "interleave") == "interleave"
    groups = [list(range(0, world, 2)), list(range(1, world, 2))] if inter else None
    init_tp_group(tp_degree=world // 2, groups=groups)
    group = get_tp_group()
    mine = groups[rank % 2] if inter else list(range((rank // 4) * 4, (rank // 4) * 4 + 4))
    pos = mine.index(rank)
    # pairs inside the group: (g0, g1), (g2, g3); dist.new_group must be called by every rank
    pair_groups = {}
    for gl in ([list(range(0, world, 2)), list(range(1, world, 2))] if inter
               else [list(range(0, 4)), list(range(4, 8))]):
        for a in (0, 2):
            ranks = [gl[a], gl[a + 1]]
            pair_groups[tuple(ranks)] = dist.new_group(ranks)
    pair = next(g for r, g in pair_groups.items() if rank in r)
    g = torch.Generator().manual_seed(rank)
    xs = (torch.randn(S, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    xf = (torch.randn(4 * S, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    xh = (torch.randn(4 * S, D // 4, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    x2 = (torch.randn(2 * S, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    w = (torch.randn(D, N, generator=g) * 0.02).to(torch.bfloat16).to(dev)

    def a2a(t):
        return funcol.all_to_all_single(t, None, None, group)

    fns = {
        "ag": lambda: funcol.all_gather_tensor(xs, 0, group) * 1.0,
        "rs": lambda: funcol.reduce_scatter_tensor(xf, "sum", 0, group),
        "a2a": lambda: a2a(xh) * 1.0,
        "a2a4": lambda: (a2a(xh) + a2a(xh * 2.0) + a2a(xh * 3.0) + a2a(xh * 4.0)),
        "ag2": lambda: funcol.all_gather_tensor(xs, 0, pair) * 1.0,
        "rs2": lambda: funcol.reduce_scatter_tensor(x2, "sum", 0, pair),
        "mm": lambda: torch.matmul(xf, w),
        "ag||mm": lambda: (funcol.all_gather_tensor(xs, 0, group) * 1.0, torch.matmul(xf, w)),
    }
    names = sys.argv[1:] or list(fns)
    if rank == 0:
        print(f"groups={'interleave' if inter else 'contiguous'} NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS')!r}",
              flush=True)
    for name in names:
        f = torch.compile(fns[name], backend="neuron", dynamic=False, fullgraph=True)
        try:
            out = f()
            torch.neuron.synchronize()
        except Exception as ex:
            if rank == 0:
                print(f"{name:>7}: FAILED {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            continue
        for _ in range(3):
            f()
        torch.neuron.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = f()
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        if name == "a2a" and rank == mine[0]:
            # rows [k*S:(k+1)*S] of the output came from group member k's rows [pos*S:(pos+1)*S]
            print(f"   a2a out shape {tuple(out.shape)}", flush=True)
        if rank == 0:
            print(f"{name:>7}: {ms:.3f} ms", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
