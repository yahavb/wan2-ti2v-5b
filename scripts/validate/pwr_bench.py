"""Does a kernel run slower when more cores of the device are busy (power cap)?

Every rank runs the same independent kernel in a loop, started together after a barrier, and
reports its time per call; run with 8, 4, 2 or 1 ranks and compare.
  torchrun --nproc_per_node=<n> pwr_bench.py      Env: KERNEL (mlp | attn | cmax), REPS (200)
mlp: the SP block's FFN as one nkilib mlp call ([7856, 3072] tokens, [3072, 3584] / [3584, 3072]
weights); attn: attention_cte_lb over 6 heads x 7854 tokens x 128; cmax: nkilib's experimental
attention_const_max at the same shape and layouts as FAST_CMAX (timing only).
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
KERNEL = os.environ.get("KERNEL", "mlp")
REPS = int(os.environ.get("REPS", "200"))
bf16 = torch.bfloat16


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(rank)
    if KERNEL == "mlp":
        from nkilib.core.mlp.mlp import mlp
        from nkilib.core.utils.common_types import ActFnType
        NP, H, I = 7856, 3072, 3584
        args = [torch.randn(NP, H, generator=g).to(bf16).to(dev),
                (torch.randn(H, I, generator=g) * H ** -0.5).to(bf16).to(dev),
                (0.1 * torch.randn(1, I, generator=g)).to(bf16).to(dev),
                (torch.randn(I, H, generator=g) * I ** -0.5).to(bf16).to(dev)]

        def fn(x, wu, bu, wd):
            y = mlp(x.unsqueeze(0), wu, wu, wd, up_proj_bias_tensor=bu,
                    activation_fn=ActFnType.GELU_Tanh_Approx, skip_gate_proj=True)
            return y[0] if isinstance(y, (list, tuple)) else y
    elif KERNEL == "cmax":  # nkilib const-max attention as FAST_CMAX calls it (q, k [N, d, Sk], v [N, Sk, d])
        from nkilib.experimental.attention.attention_const_max import attention_const_max
        BS, L, D = 6, 7854, 128
        args = [(0.1 * torch.randn(BS, D, L, generator=g)).to(bf16).to(dev),
                (0.1 * torch.randn(BS, D, L, generator=g)).to(bf16).to(dev),
                torch.randn(BS, L, D, generator=g).to(bf16).to(dev)]

        def fn(q_, k_, v_):
            return attention_const_max(q_, k_, v_, 10.0, D ** -0.5)
    else:
        from kernels.attention_cte_lb import attention_cte
        BS, L, D = 6, 7854, 128
        q = torch.randn(BS, L, D, generator=g)
        k = torch.randn(BS, L, D, generator=g)
        q = (q / q.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16).to(dev)
        k = (k / k.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16).transpose(1, 2).contiguous().to(dev)
        args = [q, k, torch.randn(BS, L, D, generator=g).to(bf16).to(dev)]

        def fn(q_, k_, v_):
            return attention_cte(q_, k_, v_, scale=D ** -0.5, causal_mask=False)
    f = torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)
    for _ in range(4):
        f(*args)
    torch.neuron.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(REPS):
        f(*args)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    allms = [None] * world
    dist.all_gather_object(allms, ms)
    if rank == 0:
        print(f"[pwr] {KERNEL} ranks={world} cores={os.environ.get('NEURON_RT_VISIBLE_CORES')}: "
              f"{max(allms):.3f} ms per call (max), min {min(allms):.3f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
