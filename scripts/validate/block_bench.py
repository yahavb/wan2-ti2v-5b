"""One DiT block at the 704x544x81 shape (B=2 CFG, L=7854), TP over all ranks, with ablations.

torchrun --nproc_per_node=8 block_bench.py [variant ...]
  ref      models.fast_dit._FastBlock as used in production (SELF_IMPL / FAST_CROSS_IMPL)
  full     this file's copy of the same forward (should time like ref)
  no_ar    all-reduces removed (row-parallel outputs used as-is)      -> comm cost
  no_sa    self-attention kernel replaced by v                        -> self-attn kernel cost
  no_xa    cross-attention kernel replaced by zeros                   -> cross-attn kernel cost
  no_ew    norms/modulation/RoPE/gates removed (matmuls+attn+AR only) -> elementwise cost
  mm_only  only the 7 matmuls (+ the 3 big all-reduces)        mm_noar: matmuls only
  no_rope / no_qkn / no_ln / no_mod   drop one elementwise stage
  ar_only  3 all-reduces of a [B, L, C] bf16 tensor
  sp       _FastBlockSP (Megatron SP); output rows checked against ref
  sp_lx    _FastBlockSP with local all-head cross-attention (CFG_PAR=1 only)
  sp_a2o   sp_lx + all-to-all output + unsharded O-projection (FAST_A2A_O; CFG_PAR=1 only)
           SP timing probes: SP_NOCOMM, SP_NOQKN, SP_NOROPE, SP_NOSA, SP_NOXA (=1 each)
The ablations change the math; they are timing probes only.
Accuracy: REF32=1 adds rel_rms_vs_fp32 (the block in fp32 on the CPU, cached per input set);
XMERGE=<n> zero-pads the context after n real tokens and runs the sp_lx / sp_a2o cross-attention on
the merged keys with their score bias, as FAST_CROSS_MERGE does in the pipeline.
Env: LAYER (0), REPS (20), SELF_IMPL (cte|constmax|sdpa), FAST_CROSS_IMPL,
     CFG_PAR=1 (two TP groups of world/2, batch 1: the CFG_PARALLEL layout; ranks 0..tp-1 are the
     first group, so NEURON_RT_VISIBLE_CORES=c,c+2,..,c+1,c+3,.. puts the groups on alternating
     cores as CFG_GROUPS=interleave does), BB_TP=4 with 4 ranks (one CFG group alone),
     BB_STAGGER_MS (the second group's timed loop starts that much later), REAL_X=<file> (a saved
     real block input), SP_NOLN / SP_NOPERM (more SP timing probes)
"""
import copy
import glob
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
LAYER = int(os.environ.get("LAYER", "0"))
REPS = int(os.environ.get("REPS", "20"))
SELF_IMPL = os.environ.get("SELF_IMPL", "cte")
GRID = (21, 22, 17)
L = GRID[0] * GRID[1] * GRID[2]
B, DIM, FFN, HEADS, D, LC = 2, 3072, 14336, 24, 128, 512


def load_block(layer):
    from safetensors import safe_open
    from wan.modules.model import WanAttentionBlock
    blk = WanAttentionBlock(DIM, FFN, HEADS, qk_norm=True, cross_attn_norm=True, eps=1e-6)
    sd = {}
    pre = f"blocks.{layer}."
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "diffusion_pytorch_model*.safetensors"))):
        with safe_open(f, "pt") as st:
            for n in st.keys():
                if n.startswith(pre):
                    sd[n[len(pre):]] = st.get_tensor(n)
    missing, unexpected = blk.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return blk, missing


def rope_freqs(max_seq_len, dim, theta=10000):
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    return torch.outer(torch.arange(max_seq_len), freqs).float()


class Probe(torch.nn.Module):
    """Copy of fast_dit._FastBlock.forward with switches."""

    def __init__(self, block, nf, self_impl, ar=True, sa=True, xa=True, ew=True, attn=True,
                 rope=True, qkn=True, ln=True, mod=True):
        super().__init__()
        self.block, self.nf, self.self_impl = block, nf, self_impl
        self.ar, self.sa, self.xa, self.ew, self.attn = ar, sa, xa, ew, attn
        self.rope, self.qkn, self.ln, self.mod = rope, qkn, ln, mod

    def forward(self, x, e, cos, sin, ck, cv):
        import models.fast_dit as fd
        blk, nf = self.block, self.nf
        sa, ca = blk.self_attn, blk.cross_attn
        B_, L_, _ = x.shape
        n, d = sa.num_heads, sa.head_dim
        AR = fd._all_reduce if self.ar else (lambda t: t)

        def row(layer, h):
            out = AR(F.linear(h, layer.weight, None))
            return out + layer.bias if layer.bias is not None else out

        e = (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [t.squeeze(2) for t in e]
        LN = (lambda nrm, t: nrm(t)) if self.ln else (lambda nrm, t: t)
        MOD = fd._modulate if self.mod else (lambda t, s, c, f: t)
        GATE = fd._gate if self.mod else (lambda t, g, f: t)
        if self.ew:
            h = MOD(LN(blk.norm1, x).to(torch.bfloat16), e[0], e[1], nf)
        else:
            h = x
        q = F.linear(h, sa.q.weight, sa.q.bias)
        k = F.linear(h, sa.k.weight, sa.k.bias)
        if self.ew and self.qkn:
            if self.ar:
                q, k = fd._tp_rms_qk(q, k, sa.norm_q, sa.norm_k)
            else:
                q, k = q * sa.norm_q.weight, k * sa.norm_k.weight
        v = F.linear(h, sa.v.weight, sa.v.bias).view(B_, L_, n, d)
        q, k = q.view(B_, L_, n, d), k.view(B_, L_, n, d)
        if self.ew and self.rope:
            q, k = fd._rope(q, cos, sin), fd._rope(k, cos, sin)
        if not self.attn:
            y = v
        elif not self.sa:
            y = v + q + k
        elif self.self_impl == "cte":
            y = fd._cte(q, k, v)
        elif self.self_impl == "constmax":
            y = fd._const_max_self(q, k, v)
        else:
            y = fd._sdpa(q, k, v)
        y = row(sa.o, y.reshape(B_, L_, n * d))
        x = x + (GATE(y, e[2], nf) if self.ew else y)

        h = LN(blk.norm3, x) if self.ew else x
        q = F.linear(h, ca.q.weight, ca.q.bias)
        if self.ew and self.qkn:
            q = fd._tp_rms(q, ca.norm_q) if self.ar else q * ca.norm_q.weight
        q = q.view(B_, L_, n, d)
        if not self.attn or not self.xa:
            y = q
        elif fd.FAST_CROSS_IMPL == "cte":
            y = fd._cte(q, ck, cv)
        else:
            y = fd._sdpa(q, ck, cv)
        x = x + row(ca.o, y.reshape(B_, L_, n * d))

        h = MOD(LN(blk.norm2, x).to(torch.bfloat16), e[3], e[4], nf) if self.ew else x
        h = blk.ffn[1](F.linear(h, blk.ffn[0].weight, blk.ffn[0].bias))
        y = row(blk.ffn[2], h)
        return x + (GATE(y, e[5], nf) if self.ew else y)


def main():
    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.neuron.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_grad_enabled(False)
    dev = torch.device("neuron")
    import models.fast_dit as fd
    from models.tp_utils import init_tp_group, shard_model_tp
    torch._dynamo.config.cache_size_limit = 64

    variants = sys.argv[1:] or ["ref", "full", "no_ar", "no_sa", "no_xa", "no_ew", "mm_only", "ar_only"]
    blk, missing = load_block(LAYER)
    # REF32=1: score the SP variants against the block in fp32 on the CPU (ref_fp32.block_fwd, the
    # upstream math with exact attention), computed once by rank 0 from the unsharded weights.
    ref32 = os.environ.get("REF32", "0") == "1"
    blk32 = copy.deepcopy(blk).float() if ref32 else None

    class _M:  # minimal model stand-in for shard_model_tp
        pass
    m = _M()
    m.blocks, m.num_heads, m.parameters = [blk], HEADS, blk.parameters
    # CFG_PAR=1: two TP groups of world/2 ranks, each running one CFG branch (batch 1).
    cfg_par = os.environ.get("CFG_PAR", "0") == "1"
    # BB_TP=4 with 4 ranks and CFG_PAR=1: one CFG group alone (the other group's cores idle)
    tp = int(os.environ.get("BB_TP", "0")) or (world // 2 if cfg_par else world)
    bsz = 1 if cfg_par else B
    full_cross = copy.deepcopy(blk.cross_attn).to(torch.bfloat16).to(dev)
    full_o = copy.deepcopy(blk.self_attn.o).to(torch.bfloat16).to(dev)
    init_tp_group(tp_degree=tp)
    shard_model_tp(m, rank % tp, tp)
    blk = blk.to(torch.bfloat16).to(dev)
    if rank == 0:
        print(f"layer {LAYER} (missing keys {missing}) world={world} tp={tp} batch={bsz} self={SELF_IMPL} "
              f"cross={fd.FAST_CROSS_IMPL} L={L}", flush=True)

    g = torch.Generator().manual_seed(0)
    x = (torch.randn(bsz, L, DIM, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    e0 = (torch.randn(1, GRID[0], 6, DIM, generator=g) * 0.1).to(torch.bfloat16).to(dev)
    real = os.environ.get("REAL_X", "")  # a real residual-stream input of block LAYER (+ its em = modulation + e0)
    if real:
        rd = torch.load(real)
        x = rd["x"].view(1, L, DIM).expand(bsz, L, DIM).contiguous().to(dev)
        e0 = (rd["em"].float() - blk.modulation.detach().float().cpu()).unsqueeze(0).to(torch.bfloat16).to(dev)
    c = D // 2
    freqs = torch.cat([rope_freqs(1024, D - 4 * (D // 6)), rope_freqs(1024, 2 * (D // 6)),
                       rope_freqs(1024, 2 * (D // 6))], 1)
    assert freqs.shape[1] == c
    cos, sin = fd._rope_tables(freqs, GRID, D, dev)
    ctx = (torch.randn(bsz, LC, DIM, generator=g) * 0.5).to(torch.bfloat16)
    nreal = int(os.environ.get("XMERGE", "0"))
    if nreal:
        ctx[:, nreal:] = 0  # zero-padded text context, as the pipeline's
    ctx = ctx.to(dev)
    ck, cv = torch.compile(fd._FastCrossKV(blk), backend="neuron", dynamic=False)(ctx)
    y32 = None
    if ref32:
        tag = os.path.basename(real).replace(".pt", "") if real else "rand"
        path = f"/tmp/blk_ref32_L{LAYER}_b{bsz}_x{nreal}_{tag}.pt"
        if rank == 0 and not os.path.exists(path):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from ref_fp32 import block_fwd, lin, rms
            torch.set_num_threads(int(os.environ.get("REF32_THREADS", "64")))
            c32, ca32 = ctx.float().cpu(), blk32.cross_attn
            ck32 = rms(lin(c32, ca32.k), ca32.norm_k).view(bsz, LC, ca32.num_heads, ca32.head_dim)
            cv32 = lin(c32, ca32.v).view(bsz, LC, ca32.num_heads, ca32.head_dim)
            cos32, sin32 = fd._rope_tables(freqs, GRID, D, torch.device("cpu"))
            t0 = time.perf_counter()
            out32 = block_fwd(blk32, x.float().cpu(), e0.float().cpu(), cos32, sin32, ck32, cv32, GRID[0])
            torch.save(out32, path)
            print(f"[block_bench] fp32 CPU reference {path} in {time.perf_counter() - t0:.1f}s", flush=True)
        dist.barrier()
        y32 = torch.load(path)  # [bsz, L, DIM] fp32

    from models.tp_utils import get_tp_rank, get_tp_world_size
    base_args = (x, e0, cos, sin, ck, cv)
    sp_ref = None
    for var in variants:
        args = base_args
        if var.startswith("sp"):
            # Megatron-SP block: this rank's token shard in, shard out; checked against ref rows
            # sp: TP-sharded cross-attn | sp_lx: local all-head cross-attn (needs CFG_PAR=1)
            tpw, shard = get_tp_world_size(), get_tp_rank()
            N = bsz * L
            Np = -(-N // tpw) * tpw
            S = Np // tpw
            yr = torch.compile(fd._FastBlock(blk, GRID[0], SELF_IMPL, None), backend="neuron",
                               dynamic=False)(*base_args)
            xflat = F.pad(x.reshape(N, DIM), (0, 0, 0, Np - N))
            xs = xflat[shard * S:(shard + 1) * S].contiguous()
            oh = fd._sp_onehot(bsz, L, GRID[1] * GRID[2], GRID[0], Np, shard, tpw).to(dev)
            lx = var in ("sp_lx", "sp_a2o")
            kk, vv, cb = ck, cv, None
            cx = ctx
            if lx and nreal:  # FAST_CROSS_MERGE: n real keys + copies of the pad key, log-weighted
                lk = 128 * -(-(nreal + 1) // 128)
                idx = torch.tensor(list(range(nreal)) + [nreal] * (lk - nreal), device=ctx.device)
                cx = ctx.index_select(1, idx)
                w = torch.zeros(lk, dtype=torch.float32)
                w[nreal:] = math.log((LC - nreal) / (lk - nreal))
                cb = w.to(dev)
            if lx:
                kk, vv = torch.compile(fd._LocalCrossKV(full_cross), backend="neuron", dynamic=False)(cx)
            args = (xs, e0, oh, cos.repeat(1, bsz, 1, 1), sin.repeat(1, bsz, 1, 1), kk, vv)
            if cb is not None:
                args = args + (cb,)
            unit = fd._FastBlockSP(blk, SELF_IMPL, bsz, L, Np, full_cross=full_cross if lx else None,
                                   full_o=full_o if var == "sp_a2o" else None)
            yr = F.pad(yr.reshape(N, DIM), (0, 0, 0, Np - N))[shard * S:(shard + 1) * S].float().cpu()
            n_valid = max(0, min(S, N - shard * S))
            yr32 = None
            if y32 is not None:
                yr32 = F.pad(y32.reshape(N, DIM), (0, 0, 0, Np - N))[shard * S:(shard + 1) * S]
            sp_ref = (yr, n_valid, yr32)
        elif var == "ref":
            unit = fd._FastBlock(blk, GRID[0], SELF_IMPL, None)
        elif var == "ar_only":
            def unit(x, e, c_, s, k, v):
                for _ in range(3):
                    x = fd._all_reduce(x) * 0.125
                return x
        else:
            kw = dict(full={}, no_ar=dict(ar=False), no_sa=dict(sa=False), no_xa=dict(xa=False),
                      no_ew=dict(ew=False), mm_only=dict(ew=False, attn=False),
                      mm_noar=dict(ew=False, attn=False, ar=False),
                      no_rope=dict(rope=False), no_qkn=dict(qkn=False), no_ln=dict(ln=False),
                      no_mod=dict(mod=False))[var]
            unit = Probe(blk, GRID[0], SELF_IMPL, **kw)
        if var.startswith("sp"):
            # SP timing probes (change the math): SP_NOCOMM=1 replaces the collectives with
            # same-shape local ops; SP_NOQKN=1 / SP_NOROPE=1 drop QK-norm / RoPE.
            tpw_ = get_tp_world_size()
            if os.environ.get("SP_NOCOMM", "0") == "1":
                # reduce-scatter -> local sum of the row blocks: reads every row, so the compiler
                # cannot shrink the matmuls in front of it (a plain row slice let it skip 3/4 of
                # the O-projection and FFN rows)
                fd._ag = lambda t: t.repeat(tpw_, *([1] * (t.dim() - 1)))
                fd._rs = lambda t: t.view(tpw_, t.shape[0] // tpw_, *t.shape[1:]).sum(0)
                fd._all_reduce = lambda t: t
            if os.environ.get("SP_NOQKN", "0") == "1":
                fd._tp_rms_qk = lambda q, k, nq, nk: (q, k)
                fd._local_rms = lambda t, nrm: t
            if os.environ.get("SP_NOROPE", "0") == "1":
                fd._rope = lambda t, c_, s_: t
            if os.environ.get("SP_NOLN", "0") == "1":  # norm1 / norm2 / norm3 -> identity
                class _Id(torch.nn.Module):
                    def forward(self, t):
                        return t
                blk.norm1, blk.norm2, blk.norm3 = _Id(), _Id(), _Id()
            # SP_NOPERM=1: self-attention gets q / k / v as plain reshapes of the [L, n, d] tensors
            # (no head-major permute copies, no k transpose, no output transpose): the layout cost
            if os.environ.get("SP_NOPERM", "0") == "1":
                _cte_real = fd._cte

                def _cte_noperm(q, k, v, _o=_cte_real):
                    from nkilib.core.attention.attention_cte import attention_cte
                    B_, Lq, n_, d_ = q.shape
                    if k.shape[1] != Lq:  # cross-attention: unchanged
                        return _o(q, k, v)
                    out = attention_cte(q.reshape(B_ * n_, Lq, d_), k.reshape(B_ * n_, d_, Lq),
                                        v.reshape(B_ * n_, Lq, d_), scale=d_ ** -0.5, causal_mask=False)
                    return out.view(B_, Lq, n_, d_)
                fd._cte = _cte_noperm
            # SP_NOSA=1 / SP_NOXA=1: drop the self- / cross-attention kernel. Self returns v + q + k
            # so the q/k projections, QK-norm and RoPE stay live; cross returns q.
            nosa, noxa = os.environ.get("SP_NOSA", "0") == "1", os.environ.get("SP_NOXA", "0") == "1"
            if nosa or noxa:
                _cte0 = fd._cte

                def _cte_probe(q, k, v, _o=_cte0):
                    if k.shape[1] == q.shape[1]:  # self-attention: Lk == Lq
                        return v + q + k if nosa else _o(q, k, v)
                    return q if noxa else _o(q, k, v)
                fd._cte = _cte_probe
        f = torch.compile(unit, backend="neuron", dynamic=False, fullgraph=True)
        t0 = time.perf_counter()
        try:
            y = f(*args)
            torch.neuron.synchronize()
        except Exception as ex:
            if rank == 0:
                print(f"{var:>8}: FAILED {type(ex).__name__}: {str(ex)[:600]}", flush=True)
            continue
        t_first = time.perf_counter() - t0
        for _ in range(3):
            y = f(*args)
        torch.neuron.synchronize()
        dist.barrier()
        # BB_STAGGER_MS=<ms> (CFG_PAR=1): the second TP group starts its timed loop that much later,
        # so the two groups run the same block phase-shifted (they never sync with each other here)
        stag = float(os.environ.get("BB_STAGGER_MS", "0")) / 1e3 if rank >= tp else 0.0
        if stag:
            time.sleep(stag)
        t0 = time.perf_counter()
        for _ in range(REPS):
            y = f(*args)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        if os.environ.get("BB_STAGGER_MS") and world > tp:
            allms = [None] * world
            dist.all_gather_object(allms, ms)
            if rank == 0:
                print(f"{var:>8}: stagger {os.environ['BB_STAGGER_MS']} ms: group 0 "
                      f"{max(allms[:tp]):.2f} ms/block, group 1 {max(allms[tp:]):.2f} ms/block", flush=True)
        chk = ""
        if var.startswith("sp") and sp_ref is not None:
            yr, nv, yr32 = sp_ref
            ys = y.float().cpu()[:nv]
            chk = (f" rel_rms_vs_ref_rows={((ys - yr[:nv]).pow(2).mean().sqrt() / yr[:nv].pow(2).mean().sqrt()).item():.5f}"
                   f" pad_rows_finite={bool(torch.isfinite(y.float().cpu()[nv:]).all())}")
            if yr32 is not None:  # this rank's rows vs the fp32 CPU block
                r32 = yr32[:nv]
                chk += f" rel_rms_vs_fp32={((ys - r32).pow(2).mean().sqrt() / r32.pow(2).mean().sqrt()).item():.5f}"
        if rank == 0:
            print(f"{var:>8}: {ms:7.2f} ms/block  (first call {t_first:.1f}s) "
                  f"out_rms={y.float().pow(2).mean().sqrt().item():.4f}{chk}", flush=True)
        elif chk and os.environ.get("ALL_RANKS", "0") == "1":  # every shard of both CFG groups
            print(f"{var:>8} rank {rank}:{chk}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
