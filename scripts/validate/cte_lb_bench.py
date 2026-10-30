"""nkilib attention_cte vs the lingbot-world vendored copy (kernels/attention_cte_lb.py, with its
LINGBOT_CTE_* data-movement knobs) at this pipeline's self-attention call: CFG-parallel TP4, one
call per rank over 6 heads x 7854 tokens x 128, non-causal, q (bs, Lq, d) / k (bs, d, Lk) / v.
Run on every rank at once (the DiT runs all 8 cores together), each call inside its own compiled
graph as in the block. Prints the time (max over ranks) and saves rank 0's output to
$OUT_DIR/<TAG>.pt for bitwise comparison between variants (CMP=<tag> compares against it).
torchrun --nproc_per_node=8 cte_lb_bench.py   Env: KERNEL (nkilib | lb), TAG, REPS (20), OUT_DIR,
CMP, and the lb knobs LINGBOT_CTE_VGROUP / KGROUP / KV_DGE / KV_DBUF. Use NKI_ENABLE_TRACE_CACHE=0:
the persistent trace cache ignores the knobs (they are read at import, not kernel arguments).
"""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root

KERNEL = os.environ.get("KERNEL", "nkilib")
TAG = os.environ.get("TAG", KERNEL)
REPS = int(os.environ.get("REPS", "20"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cte_lb_bench")
CMP = os.environ.get("CMP", "")
BS, L, D = 6, 7854, 128


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    if KERNEL == "lb":
        from kernels.attention_cte_lb import attention_cte
    else:
        from nkilib.core.attention.attention_cte import attention_cte
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(rank)
    # q/k with unit-RMS rows times 1.5 (scores of a few units, like the normed DiT q/k), v random
    q = torch.randn(BS, L, D, generator=g)
    k = torch.randn(BS, L, D, generator=g)
    q = (q / q.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(torch.bfloat16).to(dev)
    k = (k / k.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(torch.bfloat16)
    kn = k.transpose(1, 2).contiguous().to(dev)  # (bs, d, Lk): tp_k=False
    v = torch.randn(BS, L, D, generator=g).to(torch.bfloat16).to(dev)

    def attn(q_, k_, v_):
        return attention_cte(q_, k_, v_, scale=D ** -0.5, causal_mask=False)
    f = torch.compile(attn, backend="neuron", dynamic=False, fullgraph=True)
    t0 = time.perf_counter()
    out = f(q, kn, v)
    torch.neuron.synchronize()
    t_first = time.perf_counter() - t0
    for _ in range(3):
        out = f(q, kn, v)
    torch.neuron.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(q, kn, v)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    allms = [None] * world
    dist.all_gather_object(allms, ms)
    if rank == 0:
        o = out.float().cpu()
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(o, os.path.join(OUT_DIR, f"{TAG}.pt"))
        cmp = ""
        if CMP and os.path.exists(os.path.join(OUT_DIR, f"{CMP}.pt")):
            r = torch.load(os.path.join(OUT_DIR, f"{CMP}.pt"))
            dif = (o - r).abs()
            cmp = (f"; vs {CMP}: bitwise equal {torch.equal(o, r)}, max |diff| {dif.max().item():.3e}, "
                   f"rel rms {(dif.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item():.2e}")
        knobs = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("LINGBOT_CTE"))
        print(f"[cte_lb] {TAG} ({KERNEL} {knobs}): {max(allms):.3f} ms per call (max over {world} ranks, "
              f"min {min(allms):.3f}; first call {t_first:.1f} s){cmp}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
