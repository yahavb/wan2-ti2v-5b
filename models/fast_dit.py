"""Fast DiT execution path for Wan2.2-TI2V-5B on Neuron (TP only, SP=1).

Ports the main Difflet ideas onto our TP-sharded WanModel without changing weights:

  1. Whole-block compilation: each WanAttentionBlock (norms, modulation, QKV, RoPE,
     attention, O-proj, cross-attn, FFN) is ONE torch.compile graph with the TP
     all-reduces inside the graph (functional collectives), instead of the old
     eager ops + per-FFN NEFFs + @torch.compiler.disable all-reduces.
  2. Batched CFG: cond and uncond run as batch=2 in one forward (the latent and
     timestep are shared; only the text context differs).
  3. RoPE cos/sin built once per grid on device (the eager path rebuilt them on CPU
     and copied host->device twice per block per forward).
  4. Per-frame modulation: TI2V timesteps are per-token, but only two distinct values
     exist (0 for the image frame, t elsewhere), constant within a latent frame. The
     time embedding / 6-way modulation is computed for F frames instead of L tokens
     (21 vs 8190 rows) and broadcast over each frame's tokens.
  5. Text embedding and every block's cross-attention K/V (incl. k-norm) computed once
     per generation (context is fixed across steps).

Attention backend (FAST_ATTN_IMPL):
  sdpa (default) -- F.scaled_dot_product_attention; TorchNeuron lowers it to its
                    built-in NKI flash kernel, LNC2-aware (splits heads across both
                    physical cores).
  nki            -- our kernels/ NST self-attn + cross-attn (single-program launch,
                    so under LNC2 they occupy one physical core).
  constmax       -- self-attn via kernels/attention_const_max.py (odyssey-trainium
                    const-max kernel, constant softmax max CONST_MAX_SOFTMAX_MAX);
                    cross-attn stays on sdpa.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol

from models.tp_utils import get_tp_group, get_tp_world_size

logger = logging.getLogger(__name__)

FAST_ATTN_IMPL = os.environ.get("FAST_ATTN_IMPL", "sdpa")
# Cross-attention backend for the non-"nki" impls: sdpa (default) or cte (nkilib attention_cte).
FAST_CROSS_IMPL = os.environ.get("FAST_CROSS_IMPL", "sdpa")
FAST_FULLGRAPH = os.environ.get("FAST_FULLGRAPH", "0") == "1"
# FAST_SP=1: Megatron-style sequence parallelism inside the TP group (_FastBlockSP): token-
# sharded residual stream, all-gather before / reduce-scatter after the TP matmuls.
FAST_SP = os.environ.get("FAST_SP", "0") == "1"
# FAST_A2A_O=1 (with FAST_SP + local cross-attention): the self-attention output goes head -> token
# shard by one all-to-all, and the O-projection runs there with the unsharded weight, instead of a
# row-parallel matmul + reduce-scatter (9 MB moved per rank instead of 36 MB). The unsharded O
# copies are made by inference_neuron_ti2v.py (+0.43 GB per rank).
FAST_A2A_O = os.environ.get("FAST_A2A_O", "0") == "1"
# FAST_NKI_MLP=1 (with FAST_SP): the FFN (up + bias, GELU-tanh, down) runs as one nkilib mlp kernel
# call instead of two compiler matmuls with the GELU decomposed into ~5 ScalarE passes. The kernel
# takes [H, I] / [I, H] weights, so the block keeps pre-transposed copies (made once).
FAST_NKI_MLP = os.environ.get("FAST_NKI_MLP", "0") == "1"
# FAST_MOD_SPLIT=1 (FAST_SP): the per-token modulation is 6 one-hot matmuls [S, F] @ [F, C], one
# where each vector is used, instead of one [S, F] @ [F, 6C] product materialized as a [S, 6, C]
# tensor (72 MB per block). Same values (a one-hot row selects one row exactly).
FAST_MOD_SPLIT = os.environ.get("FAST_MOD_SPLIT", "0") == "1"
# FAST_TIMING=1: device-sync around each compiled unit and log per-block ms (diagnostic only;
# the syncs serialize host and device, so the total is a little above the untimed step).
FAST_TIMING = os.environ.get("FAST_TIMING", "0") == "1"


def _device_sync():
    for mod in (getattr(torch, "neuron", None), getattr(torch, "accelerator", None)):
        fn = getattr(mod, "synchronize", None) if mod is not None else None
        if fn is not None:
            fn()
            return
    import torch_neuronx
    torch_neuronx.synchronize()
# Constant max subtracted before exp in the const-max kernel (scaled-score units). The output
# is invariant to it; it only has to stay within ~80 of every row's true max.
CONST_MAX_SOFTMAX_MAX = float(os.environ.get("CONST_MAX_SOFTMAX_MAX", "15.0"))
# Which layers use const-max (others fall back to sdpa, as in odyssey PR #96's hybrid):
#   "auto" -- layers whose worst-case scaled score sqrt(d)*max|g_q|*max|g_k| (per head, from
#             the QK-norm weights) stays <= CONST_MAX_HARD_LIMIT; the kernel clips scaled
#             scores at 70, so layers that can exceed it would have their peaks flattened.
#   "all"  -- every layer;  or a comma list of layer indices.
CONST_MAX_LAYERS = os.environ.get("CONST_MAX_LAYERS", "auto")
# SBUF bytes the const-max kernel may use per call for Q + fp32 output accumulators.
CONST_MAX_SBUF_BUDGET = int(os.environ.get("CONST_MAX_SBUF_BUDGET", str(8 * 1024 * 1024)))
CONST_MAX_HARD_LIMIT = float(os.environ.get("CONST_MAX_HARD_LIMIT", "70"))


def const_max_layer_ok(block):
    """Worst-case scaled score of this block's self-attn heads (local shard) under the clip?"""
    sa = block.self_attn
    d = sa.head_dim
    gq = sa.norm_q.weight.detach().float().cpu().reshape(-1, d).abs().amax(1)
    gk = sa.norm_k.weight.detach().float().cpu().reshape(-1, d).abs().amax(1)
    hard = float((d ** 0.5) * (gq * gk).max())
    return hard <= CONST_MAX_HARD_LIMIT, hard
# In-graph all-reduce chunking (0 = off). The 8MB eager limit in tp_utils was an NRT
# eager-collective constraint; set this if the compiled collective hits a size error.
FAST_ALLREDUCE_CHUNK_BYTES = int(os.environ.get("FAST_ALLREDUCE_CHUNK_BYTES", "0"))


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    if get_tp_world_size() <= 1:
        return x
    group = get_tp_group()
    chunk = FAST_ALLREDUCE_CHUNK_BYTES // x.element_size()
    if chunk <= 0 or x.numel() <= chunk:
        return funcol.all_reduce(x, "sum", group)
    # Chunk along dim 1 (sequence) so each piece stays under the byte limit.
    rows = max(1, chunk // (x.numel() // x.shape[1]))
    parts = [funcol.all_reduce(p, "sum", group) for p in x.split(rows, dim=1)]
    return torch.cat(parts, dim=1)


def _mm(x, layer):
    """x @ W^T without the bias (row-parallel layers add it after the collective)."""
    return F.linear(x, layer.weight, None)


def _lin(x, layer):
    return F.linear(x, layer.weight, layer.bias)


def _ag(x):
    """All-gather token shards [S, C] -> [T*S, C] over the TP group (rank order)."""
    if get_tp_world_size() <= 1:
        return x
    return funcol.all_gather_tensor(x, 0, get_tp_group())


def _rs(x):
    """Reduce-scatter (sum) [T*S, C] -> this rank's token shard [S, C]."""
    if get_tp_world_size() <= 1:
        return x
    return funcol.reduce_scatter_tensor(x, "sum", 0, get_tp_group())


def _head2seq(x):
    """All tokens, this rank's heads [T*S, C/T] -> this rank's token shard, all heads [S, C]
    (heads in TP-rank order, as in the unsharded projection), by one all-to-all."""
    T = get_tp_world_size()
    y = funcol.all_to_all_single(x.contiguous(), None, None, get_tp_group())  # [T (head group), S, C/T]
    S = x.shape[0] // T
    return y.view(T, S, x.shape[1]).transpose(0, 1).reshape(S, T * x.shape[1])
def _row_linear(layer, x):
    """RowParallelLinear: local matmul, in-graph all-reduce, then the (full) bias."""
    out = _all_reduce(_mm(x, layer))
    if layer.bias is not None:
        out = out + layer.bias
    return out


def _tp_rms_qk(q, k, norm_q, norm_k):
    """TPRMSNorm for q and k with ONE all-reduce of both sum-of-squares.

    q [B, Lq, local], k [B, Lk, local]; Lq != Lk for cross-attn, so concat on seq.
    """
    qf, kf = q.float(), k.float()
    Lq = q.shape[1]
    ss = torch.cat([qf.pow(2).sum(-1, keepdim=True), kf.pow(2).sum(-1, keepdim=True)], dim=1)
    ss = _all_reduce(ss)
    q_inv = torch.rsqrt(ss[:, :Lq] / norm_q.global_dim + norm_q.eps)
    k_inv = torch.rsqrt(ss[:, Lq:] / norm_k.global_dim + norm_k.eps)
    q = (qf * q_inv).type_as(q) * norm_q.weight
    k = (kf * k_inv).type_as(k) * norm_k.weight
    return q, k


def _local_rms(x, norm):
    """RMSNorm over the full (unsharded) last dim, fp32 like _tp_rms (no collective)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)).type_as(x) * norm.weight


def _tp_rms(x, norm):
    """TPRMSNorm for one tensor (global RMS via all-reduce of the local sum-of-squares)."""
    xf = x.float()
    ss = _all_reduce(xf.pow(2).sum(-1, keepdim=True))
    return (xf * torch.rsqrt(ss / norm.global_dim + norm.eps)).type_as(x) * norm.weight


def _rope(x, cos, sin):
    """x [B, L, n, d]; cos/sin [1, L, 1, d/2] fp32. Interleaved-pair rotation."""
    B, L, n, d = x.shape
    xp = x.float().reshape(B, L, n, d // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)
    return out.reshape(B, L, n, d).to(x.dtype)


def _modulate(x, shift, scale, num_frames):
    """x [B, L, C]; shift/scale [1, F, C] -> broadcast over each frame's tokens."""
    B, L, C = x.shape
    xv = x.view(B, num_frames, L // num_frames, C)
    return (xv * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)).view(B, L, C)


def _gate(y, gate, num_frames):
    B, L, C = y.shape
    return (y.view(B, num_frames, L // num_frames, C) * gate.unsqueeze(2)).view(B, L, C)


# ─── Attention backends ─────────────────────────────────────────────────────

def _sdpa(q, k, v):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> [B, Lq, n, d]."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2)


def _nki_self(q, k, v):
    from wan.modules import attention as A
    B, L, n, d = q.shape
    qn = q.permute(0, 2, 3, 1).reshape(B * n, d, L)
    kn = k.permute(0, 2, 3, 1).reshape(B * n, d, L)
    vn = v.permute(0, 2, 1, 3).reshape(B * n, L, d)
    pad_q = (128 - L % 128) % 128
    pad_k = (512 - L % 512) % 512
    if pad_q:
        qn = F.pad(qn, (0, pad_q))
    if pad_k:
        kn = F.pad(kn, (0, pad_k))
        vn = F.pad(vn, (0, 0, 0, pad_k))
    out = A._nki_self_attn_nst(qn, kn, vn, softmax_scale=d ** -0.5,
                               actual_seqlen_k=L, use_dynamic_loop=True)
    return out[:L].view(L, B, n, d).permute(1, 0, 2, 3)


def _const_max_self(q, k, v):
    """q/k/v [B, L, n, d] -> [B, L, n, d] via the const-max NKI kernel (batch folded into heads)."""
    from kernels.attention_const_max import attention_const_max
    B, L, n, d = q.shape
    qn = q.permute(0, 2, 3, 1).reshape(B * n, d, L)
    kn = k.permute(0, 2, 3, 1).reshape(B * n, d, L)
    vn = v.permute(0, 2, 1, 3).reshape(B * n, L, d)
    # The kernel keeps every head's Q blocks and fp32 output accumulator resident in SBUF;
    # at long sequences that overflows SBUF (neuronx-cc sb_spill assertion), so call it on
    # as many heads at a time as fit in CONST_MAX_SBUF_BUDGET bytes.
    per_head = d * L * 2 + 128 * (-(-L // 128)) * (d + 1) * 4
    g = max(1, min(B * n, CONST_MAX_SBUF_BUDGET // per_head))
    outs = [attention_const_max(qn[i:i + g], kn[i:i + g], vn[i:i + g], CONST_MAX_SOFTMAX_MAX, d ** -0.5)
            for i in range(0, B * n, g)]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return out.view(B, n, L, d).transpose(1, 2)


def _cte(q, k, v):
    """q [B, Lq, n, d], k/v [B, Lk, n, d] -> [B, Lq, n, d] via nkilib attention_cte.

    Exact flash attention (online softmax, no score-range limit, any seqlen), non-causal.
    Batch and heads are folded into the kernel's batch axis; at LNC1 it runs on one core.
    """
    from nkilib.core.attention.attention_cte import attention_cte
    B, Lq, n, d = q.shape
    Lk = k.shape[1]
    qn = q.permute(0, 2, 1, 3).reshape(B * n, Lq, d)  # tp_q=True layout (bs, seqlen_q, d)
    kn = k.permute(0, 2, 3, 1).reshape(B * n, d, Lk)  # tp_k=False layout (bs, d, seqlen_kv)
    vn = v.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    out = attention_cte(qn, kn, vn, scale=d ** -0.5, causal_mask=False)  # (bs, Lq, d)
    return out.view(B, n, Lq, d).transpose(1, 2)


def _nki_cross(q, k, v, identity):
    from wan.modules import attention as A
    B, L, n, d = q.shape
    Lk = k.shape[1]
    qn = q.permute(0, 2, 3, 1).reshape(B * n, d, L)
    kn = k.permute(0, 2, 3, 1).reshape(B * n, d, Lk)
    vn = v.permute(0, 2, 1, 3).reshape(B * n, Lk, d)
    pad_q = (128 - L % 128) % 128
    if pad_q:
        qn = F.pad(qn, (0, pad_q))
    out = A._nki_cross_attn(qn, kn, vn, identity, softmax_scale=d ** -0.5)
    return out[:L].view(L, B, n, d).permute(1, 0, 2, 3)


# ─── Compiled units ─────────────────────────────────────────────────────────

class _FastBlock(nn.Module):
    """One WanAttentionBlock as a single graph. Mirrors WanAttentionBlock.forward."""

    def __init__(self, block, num_frames, attn_impl, identity):
        super().__init__()
        self.block = block
        self.num_frames = num_frames
        self.attn_impl = attn_impl
        self.identity = identity

    def forward(self, x, e, cos, sin, ck, cv):
        blk, nf = self.block, self.num_frames
        sa, ca = blk.self_attn, blk.cross_attn
        B, L, _ = x.shape
        n, d = sa.num_heads, sa.head_dim

        e = (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [t.squeeze(2) for t in e]  # each [1, F, C]

        # self-attention
        h = _modulate(blk.norm1(x).to(torch.bfloat16), e[0], e[1], nf)
        q, k = _tp_rms_qk(_lin(h, sa.q), _lin(h, sa.k), sa.norm_q, sa.norm_k)
        v = _lin(h, sa.v).view(B, L, n, d)
        q = _rope(q.view(B, L, n, d), cos, sin)
        k = _rope(k.view(B, L, n, d), cos, sin)
        if self.attn_impl == "nki":
            y = _nki_self(q, k, v)
        elif self.attn_impl == "constmax":
            y = _const_max_self(q, k, v)
        elif self.attn_impl == "cte":
            y = _cte(q, k, v)
        else:
            y = _sdpa(q, k, v)
        y = _row_linear(sa.o, y.reshape(B, L, n * d))
        x = x + _gate(y, e[2], nf)

        # cross-attention; ck/cv [B, Lc, n, d] precomputed once per prompt (_FastCrossKV)
        h = blk.norm3(x)
        q = _tp_rms(_lin(h, ca.q), ca.norm_q).view(B, L, n, d)
        if self.attn_impl == "nki":
            y = _nki_cross(q, ck, cv, self.identity)
        elif FAST_CROSS_IMPL == "cte":
            y = _cte(q, ck, cv)
        else:
            y = _sdpa(q, ck, cv)
        x = x + _row_linear(ca.o, y.reshape(B, L, n * d))

        # FFN
        h = _modulate(blk.norm2(x).to(torch.bfloat16), e[3], e[4], nf)
        h = blk.ffn[1](_lin(h, blk.ffn[0]))
        y = _row_linear(blk.ffn[2], h)
        return x + _gate(y, e[5], nf)


class _FastBlockSP(nn.Module):
    """One WanAttentionBlock with Megatron-style sequence parallelism inside the TP group.

    The residual stream is a token shard xs [S, C] per rank (the B*L tokens flattened and
    zero-padded to Np = T*S). Before each column-parallel projection the normed shard is
    all-gathered to [Np, C]; after each row-parallel projection the partial sums are
    reduce-scattered back to [S, C] (the same bytes as the all-reduce they replace). So the
    LayerNorms, modulation, gates and residual adds run on S = Np/T tokens instead of all of
    them; QK-norm, RoPE and attention stay on all tokens of this rank's heads.

    Per-token modulation comes from oh [S, F] (one-hot frame of each shard token; all-zero on
    pad rows): oh @ (modulation + e) selects each token's frame row exactly. Pad rows never
    reach a real token: they are dropped before attention, and every other op is token-wise.
    """

    def __init__(self, block, attn_impl, batch, seq_len, n_pad, full_cross=None, full_o=None,
                 free_unused=False):
        """full_cross: an unsharded WanCrossAttention (bf16, on device). When given (batch 1 only),
        cross-attention runs on this rank's token shard with all heads and the precomputed
        all-head K/V -- no collectives."""
        super().__init__()
        self.block = block
        self.attn_impl = attn_impl
        self.B, self.L = batch, seq_len
        self.N = batch * seq_len
        self.Np = n_pad
        assert full_cross is None or batch == 1, "local cross-attention needs batch 1 (CFG-parallel)"
        self.xca = full_cross
        self.full_o = full_o  # unsharded self-attention O-projection (FAST_A2A_O), or None
        self.mlp_w = None
        if FAST_NKI_MLP:  # nkilib mlp layouts: up [H, I] (+ bias [1, I]), down [I, H]
            f0, f2 = block.ffn[0], block.ffn[2]
            t = lambda w: w.detach().cpu().t().contiguous().to(w.device)  # noqa: E731 (transpose on CPU)
            self.mlp_w = (t(f0.weight), f0.bias.detach().view(1, -1).contiguous(), t(f2.weight))
            if free_unused:  # the FFN now reads only the transposed copies (and ffn[2].bias)
                f0.weight = nn.Parameter(f0.weight.new_empty(0), requires_grad=False)
                f2.weight = nn.Parameter(f2.weight.new_empty(0), requires_grad=False)
        if full_o is not None and free_unused:  # the sharded O-projection is unused then
            block.self_attn.o.weight = nn.Parameter(block.self_attn.o.weight.new_empty(0), requires_grad=False)

    def _self_attn(self, q, k, v):
        if self.attn_impl == "constmax":
            return _const_max_self(q, k, v)
        if self.attn_impl == "cte":
            return _cte(q, k, v)
        return _sdpa(q, k, v)

    def forward(self, xs, e, oh, cos, sin, ck, cv):
        blk = self.block
        sa, ca = blk.self_attn, blk.cross_attn
        n, d = sa.num_heads, sa.head_dim
        B, L, N, Np = self.B, self.L, self.N, self.Np
        C = xs.shape[-1]
        nf = e.shape[1]
        em = blk.modulation.unsqueeze(0) + e  # [1, F, 6, C], as in WanAttentionBlock
        if FAST_MOD_SPLIT:
            emf = em[0]  # [F, 6, C]
            mods = [torch.matmul(oh, emf[:, j]) for j in range(6)]  # 6 x [S, C]
            et = None
        else:
            et = torch.matmul(oh, em.reshape(nf, 6 * C)).view(-1, 6, C)  # [S, 6, C]
            mods = [et[:, j] for j in range(6)]

        # self-attention
        h = blk.norm1(xs).to(torch.bfloat16) * (1 + mods[1]) + mods[0]
        hf = _ag(h)  # [Np, C]
        q, k, v = _lin(hf, sa.q), _lin(hf, sa.k), _lin(hf, sa.v)
        q, k = _tp_rms_qk(q.unsqueeze(0), k.unsqueeze(0), sa.norm_q, sa.norm_k)
        q = _rope(q[:, :N].reshape(1, N, n, d), cos, sin).view(B, L, n, d)
        k = _rope(k[:, :N].reshape(1, N, n, d), cos, sin).view(B, L, n, d)
        v = v[:N].reshape(B, L, n, d)
        y = F.pad(self._self_attn(q, k, v).reshape(N, n * d), (0, 0, 0, Np - N))
        if self.full_o is not None:
            y = _lin(_head2seq(y), self.full_o)  # [S, C]: my tokens, all heads, unsharded O
        else:
            y = _rs(_mm(y, sa.o)) + sa.o.bias
        xs = xs + y * mods[2]

        # cross-attention; ck/cv precomputed once per prompt (_FastCrossKV / _LocalCrossKV)
        if self.xca is not None:
            xa = self.xca
            nh, hd = xa.num_heads, xa.head_dim
            q = _local_rms(_lin(blk.norm3(xs), xa.q), xa.norm_q).view(1, -1, nh, hd)  # [1, S, H, d]
            y = _cte(q, ck, cv) if FAST_CROSS_IMPL == "cte" else _sdpa(q, ck, cv)
            xs = xs + _lin(y.reshape(-1, nh * hd), xa.o)
        else:
            hf = _ag(blk.norm3(xs))
            q = _tp_rms(_lin(hf, ca.q).unsqueeze(0), ca.norm_q)[0, :N].view(B, L, n, d)
            y = _cte(q, ck, cv) if FAST_CROSS_IMPL == "cte" else _sdpa(q, ck, cv)
            y = F.pad(y.reshape(N, n * d), (0, 0, 0, Np - N))
            xs = xs + (_rs(_mm(y, ca.o)) + ca.o.bias)

        # FFN
        h = blk.norm2(xs).to(torch.bfloat16) * (1 + mods[4]) + mods[3]
        hf = _ag(h)
        if self.mlp_w is not None:
            from nkilib.core.mlp.mlp import mlp
            from nkilib.core.utils.common_types import ActFnType
            w_up, b_up, w_down = self.mlp_w
            y = mlp(hf.unsqueeze(0), w_up, w_up, w_down, up_proj_bias_tensor=b_up,
                    activation_fn=ActFnType.GELU_Tanh_Approx, skip_gate_proj=True)
            y = y[0] if isinstance(y, (list, tuple)) else y
            y = _rs(y.reshape(Np, C)) + blk.ffn[2].bias
        else:
            y = _rs(_mm(blk.ffn[1](_lin(hf, blk.ffn[0])), blk.ffn[2])) + blk.ffn[2].bias
        return xs + y * mods[5]


class _FastPatchSP(nn.Module):
    """Patch embedding -> this rank's token shard [S, C] of the flattened, padded [B*L, C]."""

    def __init__(self, patch_embedding, batch, n_pad, shard_index, tp):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.batch, self.n_pad, self.shard_index, self.tp = batch, n_pad, shard_index, tp

    def forward(self, latent):
        x = self.patch_embedding(latent.unsqueeze(0)).flatten(2).transpose(1, 2)  # [1, L, C]
        B, (_, L, C) = self.batch, x.shape
        x = F.pad(x.expand(B, L, C).reshape(B * L, C), (0, 0, 0, self.n_pad - B * L))
        s = self.n_pad // self.tp
        return x[self.shard_index * s:(self.shard_index + 1) * s].contiguous()


class _FastHeadSP(nn.Module):
    """SP head: norm + modulation + projection on the shard, then all-gather the (small)
    projected tokens, unpatchify and combine CFG (batch 2) or the CFG-parallel pair."""

    def __init__(self, head, grid, patch_size, out_dim, guide_scale, batch, pair_group=None):
        super().__init__()
        self.head = head
        self.grid, self.patch_size, self.out_dim = grid, patch_size, out_dim
        self.guide_scale, self.batch, self.pair_group = guide_scale, batch, pair_group

    def forward(self, xs, e, oh, w=None):
        hd = self.head
        f, h, wd = self.grid
        p, q, r = self.patch_size
        c = self.out_dim
        nf, C = e.shape[1], xs.shape[-1]
        m = hd.modulation.unsqueeze(0) + e.unsqueeze(2)  # [1, F, 2, C]
        mt = torch.matmul(oh, m.reshape(nf, 2 * C)).view(-1, 2, C)  # [S, 2, C]
        y = hd.head(hd.norm(xs) * (1 + mt[:, 1]) + mt[:, 0])  # [S, p*q*r*c]
        B = self.batch
        y = _ag(y)[:B * f * h * wd]
        y = y.view(B, f, h, wd, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        y = y.reshape(B, c, f * p, h * q, wd * r).to(torch.bfloat16)
        if w is not None:  # CFG-parallel: this group's branch, weighted, summed over the pair
            return funcol.all_reduce(y[0].float() * w, "sum", self.pair_group).to(torch.bfloat16)
        cond, uncond = y[0], y[1]
        return uncond + self.guide_scale * (cond - uncond)


def _sp_onehot(batch, seq_len, tokens_per_frame, num_frames, n_pad, shard_index, tp):
    """[S, F] bf16: row j = one-hot frame of flattened token shard_index*S + j (0 on pad rows)."""
    s = n_pad // tp
    idx = torch.arange(shard_index * s, (shard_index + 1) * s)
    valid = idx < batch * seq_len
    frame = (idx % seq_len) // tokens_per_frame
    oh = torch.zeros(s, num_frames)
    oh[valid, frame[valid]] = 1.0
    return oh.to(torch.bfloat16)


class _LocalCrossKV(nn.Module):
    """All-head cross-attention K (normed) and V from the text context with an unsharded
    WanCrossAttention; once per prompt. -> k, v [B, Lc, H, d]."""

    def __init__(self, full_cross):
        super().__init__()
        self.xca = full_cross

    def forward(self, context):
        xa = self.xca
        B, Lc, _ = context.shape
        k = _local_rms(_lin(context, xa.k), xa.norm_k).view(B, Lc, xa.num_heads, xa.head_dim)
        v = _lin(context, xa.v).view(B, Lc, xa.num_heads, xa.head_dim)
        return k, v


class _FastCrossKV(nn.Module):
    """A block's cross-attention K (normed) and V from the text context; once per prompt."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, context):
        ca = self.block.cross_attn
        B, Lc, _ = context.shape
        n, d = ca.num_heads, ca.head_dim
        k = _tp_rms(_lin(context, ca.k), ca.norm_k).view(B, Lc, n, d)
        v = _lin(context, ca.v).view(B, Lc, n, d)
        return k, v


class _FastHead(nn.Module):
    """Head + unpatchify + CFG combine -> [C_out, F, H, W] (batch is [cond, uncond]).
    pair_group is only used by the CFG-parallel subclass."""

    def __init__(self, head, grid, patch_size, out_dim, guide_scale, pair_group=None):
        super().__init__()
        self.head = head
        self.grid = grid
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.guide_scale = guide_scale
        self.pair_group = pair_group

    def forward(self, x, e):
        hd = self.head
        f, h, w = self.grid
        p, q, r = self.patch_size
        c = self.out_dim
        m = (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        y = hd.head(_modulate(hd.norm(x), m[0].squeeze(2), m[1].squeeze(2), f))
        B = y.shape[0]
        # einsum('fhwpqrc->cfphqwr') per batch row
        y = y.view(B, f, h, w, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        y = y.reshape(B, c, f * p, h * q, w * r).to(torch.bfloat16)
        cond, uncond = y[0], y[1]
        return uncond + self.guide_scale * (cond - uncond)


class _FastHeadCfgPar(_FastHead):
    """CFG-parallel head: this group's batch is ONE branch. Returns w * pred summed with the
    other group's term over the pair group (same TP rank in the other CFG group), in fp32:
    uncond + g * (cond - uncond) == g * cond + (1 - g) * uncond, with w = g (cond) or 1 - g.
    w is a runtime tensor so both groups compile the identical graph."""

    def forward(self, x, e, w):
        hd = self.head
        f, h, wd = self.grid
        p, q, r = self.patch_size
        c = self.out_dim
        m = (hd.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        y = hd.head(_modulate(hd.norm(x), m[0].squeeze(2), m[1].squeeze(2), f))
        y = y.view(1, f, h, wd, p, q, r, c).permute(0, 7, 1, 4, 2, 5, 3, 6)
        y = y.reshape(c, f * p, h * q, wd * r).to(torch.bfloat16)
        out = funcol.all_reduce(y.float() * w, "sum", self.pair_group)
        return out.to(torch.bfloat16)


class _FastPatch(nn.Module):
    def __init__(self, patch_embedding, batch):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.batch = batch

    def forward(self, latent):
        x = self.patch_embedding(latent.unsqueeze(0))  # [1, C, f, h, w]
        x = x.flatten(2).transpose(1, 2)  # [1, L, C]
        return x.expand(self.batch, -1, -1).contiguous()


class _FastText(nn.Module):
    def __init__(self, text_embedding):
        super().__init__()
        self.text_embedding = text_embedding

    def forward(self, ctx):
        return self.text_embedding(ctx)


def _rope_tables(freqs, grid, head_dim, device):
    """Same angle grid as rope_apply_neuron, built once. -> cos/sin [1, L, 1, d/2]."""
    f, h, w = grid
    c = head_dim // 2
    fr = freqs.cpu().float().split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    ang = torch.cat([
        fr[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(1, f * h * w, 1, c)
    return torch.cos(ang).to(device), torch.sin(ang).to(device)


class FastWanDiT:
    """Batched-CFG, block-compiled forward over a TP-sharded WanModel.

    Usage:
        fast = FastWanDiT(model, grid=(F, H/2, W/2), guide_scale=..., device=dev)
        fast.set_context(ctx_cond, ctx_null)    # [1, 512, 4096] each, once per prompt
        noise_pred = fast(latent, t)            # latent [C, F, H, W], t scalar tensor
    """

    def __init__(self, model, grid, guide_scale, device, cfg_group=None, pair_group=None, full_cross=None,
                 full_o=None):
        """cfg_group=None: batched CFG (batch 2 = [cond, uncond]) over one TP group.
        cfg_group=0/1 (CFG-parallel): this rank's TP group runs only the cond (0) or uncond (1)
        branch at batch 1; pair_group links it to the same TP rank of the other group, and the
        head sums the two weighted branches over it."""
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.model = model
        self.cfg_group = cfg_group
        self.grid = tuple(int(g) for g in grid)
        self.num_frames = self.grid[0]
        self.seq_len = self.grid[0] * self.grid[1] * self.grid[2]
        self.device = device
        head_dim = model.dim // model.num_heads
        self.cos, self.sin = _rope_tables(model.freqs, self.grid, head_dim, device)
        identity = torch.eye(128, dtype=torch.bfloat16, device=device) if FAST_ATTN_IMPL == "nki" else None

        kw = dict(backend="neuron", dynamic=False, fullgraph=FAST_FULLGRAPH)
        batch = 2 if cfg_group is None else 1
        self.patch = torch.compile(_FastPatch(model.patch_embedding, batch=batch), **kw)  # SP: replaced below
        self.text = torch.compile(_FastText(model.text_embedding), **kw)
        impls = [FAST_ATTN_IMPL] * len(model.blocks)
        if FAST_ATTN_IMPL == "cte":
            logger.info(f"[fast_dit] self-attn: nkilib attention_cte on all {len(impls)} layers")
        if FAST_ATTN_IMPL == "constmax":
            if CONST_MAX_LAYERS == "all":
                chosen = set(range(len(model.blocks)))
            elif CONST_MAX_LAYERS == "auto":
                # Each rank only sees its heads' gammas; take the max over ranks so every
                # rank compiles the SAME graph per block (collectives need identical NEFFs).
                hard = torch.tensor([const_max_layer_ok(b)[1] for b in model.blocks])
                if torch.distributed.is_initialized():
                    import torch.distributed as dist
                    gloo = dist.new_group(backend="gloo")
                    dist.all_reduce(hard, op=dist.ReduceOp.MAX, group=gloo)
                chosen = {i for i, h in enumerate(hard.tolist()) if h <= CONST_MAX_HARD_LIMIT}
                logger.info(f"[fast_dit] const-max hard bounds (max over ranks): "
                            f"{[round(h, 1) for h in hard.tolist()]}")
            else:
                chosen = {int(x) for x in CONST_MAX_LAYERS.split(",") if x.strip()}
            impls = ["constmax" if i in chosen else "sdpa" for i in range(len(model.blocks))]
            logger.info(f"[fast_dit] const-max on {len(chosen)}/{len(impls)} layers: {sorted(chosen)}")
        self.impls = impls
        self.sp = FAST_SP and get_tp_world_size() > 1 and FAST_ATTN_IMPL != "nki"
        self.cross_kv = [torch.compile(_FastCrossKV(b), **kw) for b in model.blocks]
        if cfg_group is None:
            self.cfg_w = None
        else:
            w = guide_scale if cfg_group == 0 else 1.0 - guide_scale
            self.cfg_w = torch.tensor(w, dtype=torch.float32).to(device)
        if self.sp:
            from models.tp_utils import get_tp_rank
            tp, shard = get_tp_world_size(), get_tp_rank()
            n_tok = batch * self.seq_len
            n_pad = -(-n_tok // tp) * tp
            tpf = self.grid[1] * self.grid[2]
            self.oh = _sp_onehot(batch, self.seq_len, tpf, self.num_frames, n_pad, shard, tp).to(device)
            self.cos = self.cos.repeat(1, batch, 1, 1)  # rope tables over the flattened B*L tokens
            self.sin = self.sin.repeat(1, batch, 1, 1)
            self.patch = torch.compile(_FastPatchSP(model.patch_embedding, batch, n_pad, shard, tp), **kw)
            local_x = full_cross is not None and batch == 1
            self.blocks = [
                torch.compile(_FastBlockSP(b, impl, batch, self.seq_len, n_pad,
                                           full_cross=full_cross[i] if local_x else None,
                                           full_o=full_o[i] if (full_o is not None and batch == 1) else None,
                                           free_unused=True),
                              **kw)
                for i, (b, impl) in enumerate(zip(model.blocks, impls))]
            if full_o is not None and batch == 1:
                logger.info("[fast_dit] self-attention output: all-to-all + unsharded O-projection")
            if FAST_NKI_MLP:
                logger.info("[fast_dit] FFN: nkilib mlp kernel (up + GELU-tanh + down in one call)")
            if local_x:
                self.cross_kv = [torch.compile(_LocalCrossKV(fc), **kw) for fc in full_cross]
                logger.info("[fast_dit] cross-attention: local on the token shard, all heads (no collectives)")
            self.head = torch.compile(_FastHeadSP(model.head, self.grid, model.patch_size, model.out_dim,
                                                  guide_scale, batch, pair_group=pair_group), **kw)
        else:
            self.oh = None
            self.blocks = [
                torch.compile(_FastBlock(b, self.num_frames, impl, identity), **kw)
                for b, impl in zip(model.blocks, impls)
            ]
            if cfg_group is None:
                self.head = torch.compile(
                    _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **kw)
            else:
                self.head = torch.compile(
                    _FastHeadCfgPar(model.head, self.grid, model.patch_size, model.out_dim, guide_scale,
                                    pair_group=pair_group), **kw)
        self.context = None
        cfg = "batch2" if cfg_group is None else f"parallel(group {cfg_group})"
        logger.info(f"[fast_dit] grid={self.grid} L={self.seq_len} attn={FAST_ATTN_IMPL} "
                    f"cross={FAST_CROSS_IMPL} fullgraph={FAST_FULLGRAPH} blocks={len(self.blocks)} "
                    f"cfg={cfg} seq_parallel={self.sp}")

    def set_context(self, ctx_cond, ctx_null):
        if self.cfg_group is None:
            ctx = torch.cat([ctx_cond, ctx_null], dim=0)  # [2, 512, 4096]
        else:
            ctx = ctx_cond if self.cfg_group == 0 else ctx_null  # [1, 512, 4096]
        context = self.text(ctx.to(torch.bfloat16))
        self.context = [kv(context) for kv in self.cross_kv]  # per block (k, v)

    def _head(self, x, e):
        if self.sp:
            return self.head(x, e, self.oh, self.cfg_w)
        return self.head(x, e) if self.cfg_w is None else self.head(x, e, self.cfg_w)

    def _block(self, i, x, e0):
        ck, cv = self.context[i]
        if self.sp:
            return self.blocks[i](x, e0, self.oh, self.cos, self.sin, ck, cv)
        return self.blocks[i](x, e0, self.cos, self.sin, ck, cv)

    def _time_embed(self, t):
        """Per-frame time embedding: frame 0 (image) has t=0, other frames t."""
        from wan.modules.model import sinusoidal_embedding_1d
        m = self.model
        t_frames = torch.full((self.num_frames,), float(t), dtype=torch.float64)
        t_frames[0] = 0.0
        s = sinusoidal_embedding_1d(m.freq_dim, t_frames).to(torch.bfloat16)
        s = s.to(self.device).unsqueeze(0)  # [1, F, freq_dim]
        e = m.time_embedding(s)  # [1, F, C]
        e0 = m.time_projection(e).unflatten(2, (6, m.dim))  # [1, F, 6, C]
        return e, e0

    def __call__(self, latent, t):
        assert self.context is not None, "call set_context() first"
        if FAST_TIMING:
            return self._timed_call(latent, t)
        e, e0 = self._time_embed(t)
        x = self.patch(latent.to(torch.bfloat16))
        for i in range(len(self.blocks)):
            x = self._block(i, x, e0)
        return self._head(x, e)

    def _timed_call(self, latent, t):
        """FAST_TIMING=1: same forward, device-synced around every unit; logs per-unit ms."""
        import time
        sync = _device_sync
        sync()
        t0 = time.perf_counter()
        e, e0 = self._time_embed(t)
        sync()
        t1 = time.perf_counter()
        x = self.patch(latent.to(torch.bfloat16))
        sync()
        t2 = time.perf_counter()
        blk_ms = []
        for i in range(len(self.blocks)):
            tb = time.perf_counter()
            x = self._block(i, x, e0)
            sync()
            blk_ms.append((time.perf_counter() - tb) * 1e3)
        t3 = time.perf_counter()
        out = self._head(x, e)
        sync()
        t4 = time.perf_counter()
        self._timing_calls = getattr(self, "_timing_calls", 0) + 1
        by_impl = {}
        for impl, ms in zip(self.impls, blk_ms):
            by_impl.setdefault(impl, []).append(ms)
        summary = " ".join(f"{k}: n={len(v)} mean={sum(v) / len(v):.2f}ms sum={sum(v):.1f}ms"
                           for k, v in by_impl.items())
        logger.info(f"[fast_dit timing] call {self._timing_calls}: time_embed {1e3 * (t1 - t0):.2f}ms "
                    f"patch {1e3 * (t2 - t1):.2f}ms blocks {1e3 * (t3 - t2):.1f}ms ({summary}) "
                    f"head {1e3 * (t4 - t3):.2f}ms total {1e3 * (t4 - t0):.1f}ms | per-block "
                    f"{[round(m, 1) for m in blk_ms]}")
        return out

    @staticmethod
    def log_graph_breaks():
        from torch._dynamo.utils import counters
        breaks = dict(counters.get("graph_break", {}))
        if breaks:
            logger.warning(f"[fast_dit] graph breaks ({sum(breaks.values())}): {breaks}")
        else:
            logger.info("[fast_dit] no graph breaks")
