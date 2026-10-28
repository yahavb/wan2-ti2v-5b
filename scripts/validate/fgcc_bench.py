"""nkilib allgather_compute_matmul (fused all-gather + matmul, comm overlapped with compute) vs
in-graph all-gather followed by a matmul, at the DiT's CFG-parallel TP4 shapes.

torchrun --nproc_per_node=8 fgcc_bench.py      (two TP groups of 4: ranks 0-3, 4-7)
Env: M_LOCAL (1964,2048), NS (2304,3584 = fused qkv, ffn-up), REPS (20), FORCE_HBM (0)
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
K = 3072


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
    from nkilib.experimental.collectives.fgcc import allgather_compute_matmul
    force_hbm = os.environ.get("FORCE_HBM", "0") == "1"
    ms = [int(x) for x in os.environ.get("M_LOCAL", "1964,2048").split(",")]
    ns = [int(x) for x in os.environ.get("NS", "2304,3584").split(",")]

    def base(lhs, rhs):
        return torch.matmul(funcol.all_gather_tensor(lhs, 0, group), rhs)

    def fused(lhs, rhs):
        out = allgather_compute_matmul(lhs, rhs, tp, world // tp, force_hbm)
        return out.reshape(-1, rhs.shape[1])

    for m in ms:
        for n in ns:
            g = torch.Generator().manual_seed(1000 + rank)
            lhs = (torch.randn(m, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            rhs = (torch.randn(K, n, generator=g) * 0.02).to(torch.bfloat16).to(dev)
            res = {}
            for name, fn in (("ag+mm", base), ("fgcc", fused)):
                f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
                try:
                    t0 = time.perf_counter()
                    y = f(lhs, rhs)
                    torch.neuron.synchronize()
                    t_first = time.perf_counter() - t0
                    for _ in range(3):
                        y = f(lhs, rhs)
                    torch.neuron.synchronize()
                    dist.barrier()
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        y = f(lhs, rhs)
                    torch.neuron.synchronize()
                    res[name] = ((time.perf_counter() - t0) / REPS * 1e3, y.float().cpu(), t_first)
                except Exception as ex:  # report and continue
                    res[name] = (None, None, f"{type(ex).__name__}: {str(ex)[:500]}")
                dist.barrier()
            if rank == 0:
                line = f"m={m} N={n} (M={tp * m}):"
                for name, (ms_, y, extra) in res.items():
                    line += f"  {name} " + (f"{ms_:.3f} ms (first {extra:.1f}s)" if ms_ is not None else f"FAILED {extra}")
                if all(v[0] is not None for v in res.values()):
                    a, b = res["ag+mm"][1], res["fgcc"][1]
                    line += f"  rel_err {((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.5f}"
                print(line, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
