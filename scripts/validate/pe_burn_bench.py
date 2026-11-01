"""How much of the device's power budget does the Tensor engine alone use? Every rank runs the same
NKI kernel: REPS_K bf16 matmuls ([128, 128] stationary, [128, 512] moving) on SBUF-resident data, no
HBM traffic inside the loop, and reports its time per call; run with 1 and 8 ranks and compare
(the FFN's mlp kernel: 4.70 ms alone, 6.17 ms with 8 busy cores). DATA=zero feeds zeros instead
of random values, to see whether the power depends on the data.
  torchrun --nproc_per_node=<n> pe_burn_bench.py      Env: REPS (200), REPS_K (8192),
  DATA (rand | zero | mzero: moving zero | szero: stationary zero | corr: moving columns correlated
  0.9 | small: moving values in [0.5, 1)),
  EXTRA (none | dma: every 16 matmuls also copy 1 MB HBM -> SBUF | act: every 4 matmuls one exp of a
  [128, 512] fp32 tile on the Scalar engine), to see what else draws on the same power budget
"""
import os
import sys
import time
import torch
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "200"))
REPS_K = int(os.environ.get("REPS_K", "8192"))
DATA = os.environ.get("DATA", "rand")
EXTRA = os.environ.get("EXTRA", "none")
def make_kernel(nmm, extra):
    import nki
    import nki.isa as nisa
    import nki.language as nl

    def kernel(a, w, big):
        out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
        a_sb = nl.ndarray((128, 4, 512), dtype=a.dtype, buffer=nl.sbuf)
        w_sb = nl.ndarray((128, 4, 128), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=a)
        nisa.dma_copy(dst=w_sb, src=w)
        acc = [nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(4)]
        nbig = big.shape[0]
        for i in range(nmm):  # 4 independent accumulators so consecutive matmuls do not serialize
            nisa.nc_matmul(dst=acc[i % 4], stationary=w_sb[:, (i // 4) % 4, :], moving=a_sb[:, i % 4, :],
                           accumulate=(i >= 4))
            if extra == "dma" and i % 16 == 0:  # 1 MB from HBM (rotating source rows), discarded
                land = nl.ndarray((128, 4096), dtype=big.dtype, buffer=nl.sbuf)
                j = (i // 16) % (nbig // 128)
                nisa.dma_copy(dst=land, src=big[j * 128:(j + 1) * 128, :],
                              dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            if extra == "act" and i % 4 == 3:  # one exp per 4 matmuls, read from the PSUM accumulator
                pr = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=pr, data=acc[(i + 1) % 4], op=nl.exp, scale=1e-3)
        res = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=res, src=acc[0])
        for j in range(1, 4):
            nisa.tensor_tensor(dst=res, data1=res, data2=acc[j], op=nl.add)
        nisa.dma_copy(dst=out, src=res)
        return out

    kernel.__name__ = kernel.__qualname__ = f"pe_burn_{nmm}_{extra}"
    return nki.jit(kernel)
def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(rank)
    a, w = torch.randn(128, 4, 512, generator=g) * 0.1, torch.randn(128, 4, 128, generator=g) * 0.1
    if DATA == "zero":
        a, w = a * 0, w * 0
    elif DATA == "mzero":  # moving operand zero, stationary random
        a = a * 0
    elif DATA == "szero":  # stationary operand zero, moving random
        w = w * 0
    elif DATA == "corr":  # moving columns correlated along the stream (rho 0.9), like neighbouring tokens
        for j in range(1, 512):
            a[:, :, j] = 0.9 * a[:, :, j - 1] + (1 - 0.81) ** 0.5 * a[:, :, j]
    elif DATA == "small":  # moving operand values of one sign and exponent range (|x| in [0.5, 1))
        a = 0.5 + 0.49 * torch.rand(128, 4, 512, generator=g)
    a, w = a.to(torch.bfloat16).to(dev), w.to(torch.bfloat16).to(dev)
    big = (torch.randn(128 * 64, 4096, generator=g) * 0.1).to(torch.bfloat16).to(dev)  # 64 MB
    k = make_kernel(REPS_K, EXTRA)
    f = torch.compile(lambda a_, w_, b_: k(a_, w_, b_), backend="neuron", dynamic=False, fullgraph=True)
    for _ in range(3):
        f(a, w, big)
    torch.neuron.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(REPS):
        f(a, w, big)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    allms = [None] * world
    dist.all_gather_object(allms, ms)
    if rank == 0:
        macs = REPS_K * 128 * 128 * 512
        print(f"[pe_burn] {DATA} extra={EXTRA} ranks={world}: {max(allms):.3f} ms per call (max; min {min(allms):.3f}), "
              f"{REPS_K} matmuls, {macs / max(allms) / 1e9 * 2:.1f} TFLOP/s per core", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    main()
