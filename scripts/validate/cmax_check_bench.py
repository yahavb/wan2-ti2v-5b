"""The row-sum check of kernels/attention_cmax (fast_dit's CMAX_CHECK), at this pipeline's
self-attention call (one TP rank: 6 heads x 7854 tokens x 128, qkv [7856, 2304], one core).

For both const-max kernels the pipeline can run, the QK-fused one (FAST_CMAX_QK) and the plain one
(FAST_CMAX_FLAT), the checked kernel is compared with the unchecked one (the outputs must be bitwise
equal) and its stats with an fp64 CPU reference of every query row's softmax sum
S = sum_j exp(score_ij - c), computed from the same bf16 q / k (qk_apply's output):
  normal    random inputs, c = 15: no row outside the band
  overflow  64 tokens' q scaled up 40x (through their sums of squares), c = 15: their rows overflow
  low       c = 69: the sums of most rows sit around e^-60, the band's lower edge
  floor     c = 75: every row below the band
  nan       one token's q is NaN: its row in each head
Rows whose reference ln S is within TOL of a band edge may go either way (bf16 probabilities, the
Scalar engine's exp); every other row must agree. c is a [128, 1] tensor here (one compile for all
cases); the normal case also runs with c as a constant, the form the pipeline uses.
  NEURON_RT_VISIBLE_CORES=<core> python3 cmax_check_bench.py      Env: REPS (20)
"""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
REPS = int(os.environ.get("REPS", "20"))
H, N, NP, D, G, EPS = 6, 7854, 7856, 128, 3072, 1e-6
LO, HI, TOL = -60.0, 78.0, 0.05
bf16 = torch.bfloat16


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    HD = H * D
    qkv = torch.randn(NP, 3 * HD, generator=g).to(bf16)
    qkv[N:] = 0
    ss = torch.stack([qkv[:, :HD].float().pow(2).sum(-1), qkv[:, HD:2 * HD].float().pow(2).sum(-1)], -1) * (G / HD)
    ss[N:] = 1.0
    wq = (1 + 0.1 * torch.randn(HD, generator=g)).to(bf16)
    wk = (1 + 0.1 * torch.randn(HD, generator=g)).to(bf16)
    ang = torch.rand(N, D // 2, generator=g) * 2 * math.pi
    return qkv, ss, wq, wk, ang.cos().float(), ang.sin().float()


def ref_ln_sums(qn, kn, c):
    """qn / kn [H, d, N] bf16 (head-major) -> ln S [H, N] in fp64 (NaN where a score is NaN)."""
    out = torch.empty(H, N, dtype=torch.float64)
    sc = D ** -0.5
    for h in range(H):
        k = kn[h].double()                                       # [d, N]
        for a in range(0, N, 1024):
            s = qn[h, :, a:a + 1024].double().t() @ k * sc - c   # [rows, N]
            out[h, a:a + 1024] = torch.logsumexp(s, -1)
    return out


def per_partition(mask):
    """[H, N] bool -> [128] count per partition (query row t * 128 + p is on partition p)."""
    n_ot = -(-N // 128)
    m = torch.zeros(H, n_ot * 128, dtype=torch.int64)
    m[:, :N] = mask.long()
    return m.view(H, n_ot, 128).sum((0, 1))


def agree(st, ln_s):
    """Kernel stats [128, 3] vs the reference, per partition: the kernel's count of rows outside the
    band may differ from the reference's only by rows within TOL of an edge.
    -> (kernel fails, reference fails, rows near an edge, ok)."""
    bad = ~((ln_s >= LO) & (ln_s <= HI))                         # NaN -> bad
    near = ((ln_s - LO).abs() < TOL) | ((ln_s - HI).abs() < TOL)
    k_p = st[:, 0].round().long()
    r_p, n_p = per_partition(bad), per_partition(near)
    ok = bool(((k_p - r_p).abs() <= n_p).all())
    return int(k_p.sum()), int(r_p.sum()), int(n_p.sum()), ok


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
    ms = (time.perf_counter() - t0) / REPS * 1e3
    return ms, (tuple(t.cpu() for t in out) if isinstance(out, (tuple, list)) else out.cpu())


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    kw = dict(backend="neuron", dynamic=False, fullgraph=True)
    from kernels.attention_cmax import attention_cmax, attention_cmax_checked, attention_cmax_qk_kernel
    from kernels.qk_prep import qk_apply_kernel
    k_qk = qk_apply_kernel(D, EPS, G, fused=True, q_t=True)
    k_fused, k_fused_chk = attention_cmax_qk_kernel(EPS, G), attention_cmax_qk_kernel(EPS, G, check=(LO, HI))
    k_flat_chk = attention_cmax_checked(LO, HI)
    sc = D ** -0.5
    prep = torch.compile(lambda *a: k_qk(*a), **kw)
    fns = {  # name -> (compiled fn, takes qkv inputs (True) or qn / kn / vn (False), checked)
        "fused": (torch.compile(lambda qkv, ss, wq, wk, cs, sn, c: k_fused(qkv, ss, wq, wk, cs, sn, c, sc), **kw), True, False),
        "fused_chk": (torch.compile(lambda qkv, ss, wq, wk, cs, sn, c: k_fused_chk(qkv, ss, wq, wk, cs, sn, c, sc), **kw), True, True),
        "flat": (torch.compile(lambda q, k, v, c: attention_cmax(q, k, v, c, sc), **kw), False, False),
        "flat_chk": (torch.compile(lambda q, k, v, c: k_flat_chk(q, k, v, c, sc), **kw), False, True),
    }
    base = make_inputs()
    cases = [("normal", 15.0, None, None), ("overflow", 15.0, 40.0, None), ("low", 69.0, None, None),
             ("floor", 75.0, None, None), ("nan", 15.0, None, 777)]
    all_ok = True
    for name, c, up, nan_tok in cases:
        qkv, ss, wq, wk, cs, sn = (t.clone() for t in base)
        if up is not None:      # 64 tokens' q scaled up: their sums of squares divided by up^2
            ss[1000:1064, 0] /= up * up
        if nan_tok is not None:
            qkv[nan_tok, :H * D] = float("nan")
        a_qkv = [t.to(dev) for t in (qkv, ss, wq, wk, cs, sn)]
        qn, kn, vn = prep(*a_qkv)
        c_t = torch.full((128, 1), c, dtype=torch.float32).to(dev)
        ln_s = ref_ln_sums(qn.cpu(), kn.cpu(), c)
        res, times = {}, {}
        for nm, (f, from_qkv, _) in fns.items():
            args = (*a_qkv, c_t) if from_qkv else (qn, kn, vn, c_t)
            times[nm], res[nm] = timed(f, args)
        line = [f"[cmax_check] {name:<8} c={c:g}: ref ln S [{ln_s.nan_to_num(nan=0).min():.1f}, "
                f"{ln_s.nan_to_num(nan=0).max():.1f}]"]
        for fam in ("fused", "flat"):
            o_ref, (o_chk, st) = res[fam], res[fam + "_chk"]
            bitwise = torch.equal(o_ref.view(torch.int16), o_chk.view(torch.int16))
            k_fail, r_fail, n_near, ok = agree(st, ln_s)
            if name == "normal":  # the kernel's smallest / largest sum vs the reference's
                ok &= abs(math.log(st[:, 1].min().item()) - ln_s.min().item()) < 0.02
                ok &= abs(math.log(st[:, 2].max().item()) - ln_s.max().item()) < 0.02
            all_ok &= bitwise and ok
            line.append(f"{fam}: out bitwise={bitwise} fails kernel {k_fail} ref {r_fail} (near edge {n_near}) "
                        f"{'OK' if ok else 'MISMATCH'}, kernel ln S [{math.log(max(st[:, 1].min().item(), 1e-45)):.1f}, "
                        f"{math.log(max(st[:, 2].max().item(), 1e-45)):.1f}], {times[fam]:.3f} -> {times[fam + '_chk']:.3f} ms")
        print(" | ".join(line), flush=True)
    # the pipeline's form: c as a constant in the graph
    a_qkv = [t.to(dev) for t in base]
    f_const = torch.compile(lambda *a: k_fused(*a, 15.0, sc), **kw)
    f_const_chk = torch.compile(lambda *a: k_fused_chk(*a, 15.0, sc), **kw)
    t0, o0 = timed(f_const, a_qkv)
    t1, (o1, st1) = timed(f_const_chk, a_qkv)
    bitwise = torch.equal(o0.view(torch.int16), o1.view(torch.int16))
    all_ok &= bitwise and int(st1[:, 0].sum().item()) == 0
    print(f"[cmax_check] constant c=15 (pipeline form): out bitwise={bitwise}, fails {int(st1[:, 0].sum().item())}, "
          f"{t0:.3f} -> {t1:.3f} ms", flush=True)
    print(f"[cmax_check] {'ALL OK' if all_ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
