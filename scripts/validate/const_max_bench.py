"""nkilib's experimental attention_const_max (no online max, no P transpose) vs the vendored
attention_cte_lb at this pipeline's self-attention call (6 heads x 7854 tokens x 128, one core):
const-max with q head-major and K / V zero-padded to 7936 keys (each K pad row adds
exp(-softmax_max) to the denominator); with the 7854 keys as they are (the kernel handles the
partial last key tile), q seq-major and q head-major (as FAST_CMAX calls it). Prints time per call
and the rel-RMS of each against an fp32 CPU softmax on head 0.
NEURON_RT_VISIBLE_CORES=<core> python3 const_max_bench.py   Env: REPS (20), SMAX (10.0)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
SMAX = float(os.environ.get("SMAX", "10.0"))
BS, L, D = 6, 7854, 128
LK = -(-L // 128) * 128
bf16 = torch.bfloat16


def timed(f, args):
    out = f(*args)
    torch.neuron.synchronize()
    for _ in range(3):
        out = f(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(*args)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    q = torch.randn(BS, L, D, generator=g)
    k = torch.randn(BS, L, D, generator=g)
    q = (q / q.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16)
    k = (k / k.pow(2).mean(-1, keepdim=True).sqrt() * 1.5).to(bf16)
    v = torch.randn(BS, L, D, generator=g).to(bf16)
    s = q[0].float() @ k[0].float().t() * D ** -0.5
    ref = torch.softmax(s, -1) @ v[0].float()  # head 0, fp32
    rel = lambda o: ((o.float().cpu() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()  # noqa: E731
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)

    from kernels.attention_cte_lb import attention_cte
    f_cte = torch.compile(lambda q_, k_, v_: attention_cte(q_, k_, v_, scale=D ** -0.5, causal_mask=False), **kw)
    t_cte, o_cte = timed(f_cte, [q.to(dev), k.transpose(1, 2).contiguous().to(dev), v.to(dev)])

    from nkilib.core.utils.common_types import AttentionLayout
    from nkilib.experimental.attention.attention_const_max import attention_const_max
    kp = torch.zeros(BS, LK, D, dtype=bf16)
    kp[:, :L] = k
    vp = torch.zeros(BS, LK, D, dtype=bf16)
    vp[:, :L] = v
    f_cm = torch.compile(lambda q_, k_, v_: attention_const_max(q_, k_, v_, SMAX, D ** -0.5), **kw)
    t_cm, o_cm = timed(f_cm, [q.transpose(1, 2).contiguous().to(dev), kp.transpose(1, 2).contiguous().to(dev),
                              vp.to(dev)])
    # q seq-major [N, Sq, d] (attention_cte's layout; the kernel DMA-transposes it), Sk = 7854
    f_cs = torch.compile(lambda q_, k_, v_: attention_const_max(
        q_, k_, v_, SMAX, D ** -0.5, q_layout=AttentionLayout.SEQ_MAJOR, k_layout=AttentionLayout.HEAD_MAJOR), **kw)
    t_cs, o_cs = timed(f_cs, [q.to(dev), k.transpose(1, 2).contiguous().to(dev), v.to(dev)])
    # as FAST_CMAX calls it: q head-major [N, d, Sq] (qk_apply q_t), Sk = 7854
    t_ch, o_ch = timed(f_cm, [q.transpose(1, 2).contiguous().to(dev), k.transpose(1, 2).contiguous().to(dev),
                              v.to(dev)])
    print(f"[const_max] {BS} x {L} x {D} one core: attention_cte_lb {t_cte:.3f} ms (rel rms {rel(o_cte[0]):.2e}); "
          f"nkilib attention_const_max, max {SMAX}: q [N, d, Sq] + K/V padded to {LK} {t_cm:.3f} ms (rel rms "
          f"{rel(o_cm[0]):.2e}), q [N, Sq, d] + {L} keys {t_cs:.3f} ms (rel rms {rel(o_cs[0]):.2e}), "
          f"q [N, d, Sq] + {L} keys {t_ch:.3f} ms (rel rms {rel(o_ch[0]):.2e})", flush=True)


if __name__ == "__main__":
    main()
