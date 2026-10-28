"""Does the compiler overlap an in-graph collective with independent compute?

torchrun --nproc_per_node=8 overlap_bench.py   (two TP groups of 4: ranks 0-3, 4-7)
  mm      matmul [M, K] @ [K, N] on a local full-size input
  ag+mm   all-gather [m, K] -> [M, K], then the same matmul
  both    ag+mm on x1 AND an independent mm on x2 in ONE graph
If both ~= (ag+mm) + mm the collective is serialized with compute; if both ~= max(...) it overlaps.
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
m, K, N = 1964, 3072, 3584


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    dev = torch.device("neuron")
    from models.tp_utils import init_tp_group, get_tp_group
    tp = world // 2
    init_tp_group(tp_degree=tp)
    group = get_tp_group()
    g = torch.Generator().manual_seed(rank)
    x1 = (torch.randn(m, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    x2 = (torch.randn(m * tp, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    w = (torch.randn(K, N, generator=g) * 0.02).to(torch.bfloat16).to(dev)

    fns = {
        "mm": lambda a, b, w: torch.matmul(b, w),
        "ag+mm": lambda a, b, w: torch.matmul(funcol.all_gather_tensor(a, 0, group), w),
        "both": lambda a, b, w: (torch.matmul(funcol.all_gather_tensor(a, 0, group), w), torch.matmul(b, w)),
        "ag": lambda a, b, w: funcol.all_gather_tensor(a, 0, group) * 1.0,
        "rs": lambda a, b, w: funcol.reduce_scatter_tensor(b, "sum", 0, group),
        "rs||mm": lambda a, b, w: (funcol.reduce_scatter_tensor(b, "sum", 0, group), torch.matmul(b, w)),
    }
    for name, fn in fns.items():
        f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
        f(x1, x2, w)
        torch.neuron.synchronize()
        for _ in range(3):
            f(x1, x2, w)
        torch.neuron.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f(x1, x2, w)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        if rank == 0:
            print(f"{name:>7}: {ms:.3f} ms", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
