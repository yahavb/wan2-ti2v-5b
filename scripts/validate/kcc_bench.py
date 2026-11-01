"""Do collectives issued INSIDE an NKI kernel (nki.collectives) overlap the kernel's own compute?
(Compiler-level in-graph collectives do not: scripts/validate/overlap_bench.py.) Two TP groups of
4 on alternating cores, as CFG_GROUPS=interleave: every rank runs, per call,
  rs      ncc.reduce_scatter of [7856, 3072] bf16 -> [1964, 3072] (the FFN's reduce-scatter)
  ag      ncc.all_gather of [1964, 3072] -> [7856, 3072] (the FFN's / QKV's all-gather)
  mm      MMS bf16 matmuls ([128, 128] x [128, 512], SBUF-resident, no HBM traffic)
  rs+mm / ag+mm   the collective issued first, then the same matmuls, in ONE kernel
If rs+mm ~= max(rs, mm) the collective overlaps; ~= rs + mm it is serialized.
  torchrun --nproc_per_node=8 kcc_bench.py [op ...]      Env: REPS (50), MMS (4096)
"""
import os
import sys
import time
import torch
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "50"))
MMS = int(os.environ.get("MMS", "4096"))
NP, C, T = 7856, 3072, 4
def make(op, groups):
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
    from nki.collectives import ReplicaGroup

    def matmuls(a, w):
        a_sb = nl.ndarray((128, 4, 512), dtype=a.dtype, buffer=nl.sbuf)
        w_sb = nl.ndarray((128, 4, 128), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=a)
        nisa.dma_copy(dst=w_sb, src=w)
        acc = [nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(4)]
        for i in range(MMS):
            nisa.nc_matmul(dst=acc[i % 4], stationary=w_sb[:, (i // 4) % 4, :], moving=a_sb[:, i % 4, :],
                           accumulate=(i >= 4))
        res = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=res, src=acc[0])
        for j in range(1, 4):
            nisa.tensor_tensor(dst=res, data1=res, data2=acc[j], op=nl.add)
        out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=res)
        return out

    def collective(x):
        rg = ReplicaGroup(groups)
        rows = x.shape[0]
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.private_hbm)  # LNC1: srcs on private HBM
        nisa.dma_copy(dst=src, src=x)
        if op.startswith("rs"):
            dst = nl.ndarray((rows // T, C), dtype=x.dtype, buffer=nl.private_hbm)
            ncc.reduce_scatter(srcs=[src], dsts=[dst], replica_group=rg, collective_dim=0, op=nl.add)
        else:
            dst = nl.ndarray((rows * T, C), dtype=x.dtype, buffer=nl.private_hbm)
            ncc.all_gather(srcs=[src], dsts=[dst], replica_group=rg, collective_dim=0)
        out = nl.ndarray(dst.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=dst)
        return out

    if op in ("rs", "ag"):
        def kernel(x, a, w):
            return collective(x)
    elif op == "mm":
        def kernel(x, a, w):
            return matmuls(a, w)
    else:
        def kernel(x, a, w):
            o1 = collective(x)
            o2 = matmuls(a, w)
            return o1, o2
    kernel.__name__ = kernel.__qualname__ = f"kcc_{op.replace('+', '_')}_{MMS}"
    return nki.jit(kernel)
def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    groups = [list(range(0, world, 2)), list(range(1, world, 2))]
    g = torch.Generator().manual_seed(rank)
    xr = (torch.randn(NP, C, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    xg = (torch.randn(NP // T, C, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    a = (torch.randn(128, 4, 512, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    w = (torch.randn(128, 4, 128, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    ref = None
    for op in sys.argv[1:] or ["mm", "rs", "rs+mm", "ag", "ag+mm"]:
        k = make(op, groups)
        x = xr if op.startswith("rs") or op == "mm" else xg
        f = torch.compile(lambda x_, a_, w_, _k=k: _k(x_, a_, w_), backend="neuron", dynamic=False, fullgraph=True)
        try:
            out = f(x, a, w)
            torch.neuron.synchronize()
        except Exception as ex:
            if rank == 0:
                print(f"{op:>6}: FAILED {type(ex).__name__}: {str(ex)[:500]}", flush=True)
            continue
        for _ in range(3):
            f(x, a, w)
        torch.neuron.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = f(x, a, w)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        allms = [None] * world
        dist.all_gather_object(allms, ms)
        if op == "rs":  # correctness: my rows of the sum over my group
            o = out.float().cpu()
            ref = o
        if rank == 0:
            print(f"{op:>6}: {max(allms):.3f} ms per call (max over ranks; min {min(allms):.3f})", flush=True)
        dist.barrier()
    if ref is not None:
        # every rank checks its reduce-scatter rows against an in-graph torch reduce-scatter
        import torch.distributed._functional_collectives as funcol
        from models.tp_utils import init_tp_group, get_tp_group
        init_tp_group(tp_degree=world // 2, groups=groups)
        r2 = torch.compile(lambda t: funcol.reduce_scatter_tensor(t, "sum", 0, get_tp_group()), backend="neuron",
                           dynamic=False, fullgraph=True)(xr).float().cpu()
        err = (ref - r2).abs().max().item()
        errs = [None] * world
        dist.all_gather_object(errs, err)
        if rank == 0:
            print(f"    rs vs torch reduce_scatter: max |diff| over ranks {max(errs):.3e}", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    main()
