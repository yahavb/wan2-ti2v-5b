"""Do the two CFG groups slow each other's collectives down?

The pipeline runs two TP4 groups at once (cond on cores c, c+2, c+4, c+6; uncond on the odd
cores), and both issue the same all-gathers / reduce-scatters at nearly the same moment. This
times each collective at the block's shapes with one group alone and with both groups at once.

  torchrun --nproc_per_node=8 cc_contention_bench.py   (GROUPS=interleave: {0,2,4,6},{1,3,5,7})
  torchrun --nproc_per_node=4 cc_contention_bench.py   (one group; pick its cores with
                                                         NEURON_RT_VISIBLE_CORES)
Env: GROUPS (contig | interleave), REPS (20).
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
TP = 4
m, K, D = 1964, 3072, 768  # shard rows, model dim, per-rank head dims (6 x 128)


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("neuron")
    from models.tp_utils import init_tp_group, get_tp_group
    groups = None
    if os.environ.get("GROUPS", "contig") == "interleave" and world == 2 * TP:
        groups = [list(range(0, world, 2)), list(range(1, world, 2))]
    init_tp_group(tp_degree=TP, groups=groups)
    group = get_tp_group()
    g = torch.Generator().manual_seed(rank)
    xs = (torch.randn(m, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    xf = (torch.randn(m * TP, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    xh = (torch.randn(m * TP, D, generator=g) * 0.5).to(torch.bfloat16).to(dev)

    fns = {
        "ag [m,C]": (lambda: funcol.all_gather_tensor(xs, 0, group) * 1.0),
        "rs [Np,C]": (lambda: funcol.reduce_scatter_tensor(xf, "sum", 0, group)),
        "a2a [Np,D]": (lambda: funcol.all_to_all_single(xh, None, None, group) * 1.0),
        "ar [Np,2]": (lambda: funcol.all_reduce(xh[:, :2].float(), "sum", group)),
    }
    for name, fn in fns.items():
        f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
        f()
        torch.neuron.synchronize()
        for _ in range(3):
            f()
        torch.neuron.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f()
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        allms = [None] * world
        dist.all_gather_object(allms, ms)
        if rank == 0:
            print(f"[cc] world={world} groups={os.environ.get('GROUPS', 'contig')} {name:>11}: "
                  f"{max(allms):.3f} ms (min {min(allms):.3f})", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
