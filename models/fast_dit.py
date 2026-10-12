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
FAST_FULLGRAPH = os.environ.get("FAST_FULLGRAPH", "0") == "1"
# Constant max subtracted before exp in the const-max kernel (scaled-score units). The output
# is invariant to it; it only has to stay within ~80 of every row's true max.
CONST_MAX_SOFTMAX_MAX = float(os.environ.get("CONST_MAX_SOFTMAX_MAX", "15.0"))
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


def _row_linear(layer, x):
    """RowParallelLinear: local matmul, in-graph all-reduce, then the (full) bias."""
    out = _all_reduce(F.linear(x, layer.weight, None))
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
    out = attention_const_max(qn, kn, vn, CONST_MAX_SOFTMAX_MAX, d ** -0.5)
    return out.view(B, n, L, d).transpose(1, 2)


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
        q, k = _tp_rms_qk(F.linear(h, sa.q.weight, sa.q.bias),
                          F.linear(h, sa.k.weight, sa.k.bias), sa.norm_q, sa.norm_k)
        v = F.linear(h, sa.v.weight, sa.v.bias).view(B, L, n, d)
        q = _rope(q.view(B, L, n, d), cos, sin)
        k = _rope(k.view(B, L, n, d), cos, sin)
        if self.attn_impl == "nki":
            y = _nki_self(q, k, v)
        elif self.attn_impl == "constmax":
            y = _const_max_self(q, k, v)
        else:
            y = _sdpa(q, k, v)
        y = _row_linear(sa.o, y.reshape(B, L, n * d))
        x = x + _gate(y, e[2], nf)

        # cross-attention; ck/cv [B, Lc, n, d] precomputed once per prompt (_FastCrossKV)
        h = blk.norm3(x)
        q = _tp_rms(F.linear(h, ca.q.weight, ca.q.bias), ca.norm_q).view(B, L, n, d)
        y = _nki_cross(q, ck, cv, self.identity) if self.attn_impl == "nki" else _sdpa(q, ck, cv)
        x = x + _row_linear(ca.o, y.reshape(B, L, n * d))

        # FFN
        h = _modulate(blk.norm2(x).to(torch.bfloat16), e[3], e[4], nf)
        h = blk.ffn[1](F.linear(h, blk.ffn[0].weight, blk.ffn[0].bias))
        y = _row_linear(blk.ffn[2], h)
        return x + _gate(y, e[5], nf)


class _FastCrossKV(nn.Module):
    """A block's cross-attention K (normed) and V from the text context; once per prompt."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, context):
        ca = self.block.cross_attn
        B, Lc, _ = context.shape
        n, d = ca.num_heads, ca.head_dim
        k = _tp_rms(F.linear(context, ca.k.weight, ca.k.bias), ca.norm_k).view(B, Lc, n, d)
        v = F.linear(context, ca.v.weight, ca.v.bias).view(B, Lc, n, d)
        return k, v


class _FastHead(nn.Module):
    """Head + unpatchify + CFG combine -> [C_out, F, H, W]."""

    def __init__(self, head, grid, patch_size, out_dim, guide_scale):
        super().__init__()
        self.head = head
        self.grid = grid
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.guide_scale = guide_scale

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

    def __init__(self, model, grid, guide_scale, device):
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.model = model
        self.grid = tuple(int(g) for g in grid)
        self.num_frames = self.grid[0]
        self.seq_len = self.grid[0] * self.grid[1] * self.grid[2]
        self.device = device
        head_dim = model.dim // model.num_heads
        self.cos, self.sin = _rope_tables(model.freqs, self.grid, head_dim, device)
        identity = torch.eye(128, dtype=torch.bfloat16, device=device) if FAST_ATTN_IMPL == "nki" else None

        kw = dict(backend="neuron", dynamic=False, fullgraph=FAST_FULLGRAPH)
        self.patch = torch.compile(_FastPatch(model.patch_embedding, batch=2), **kw)
        self.text = torch.compile(_FastText(model.text_embedding), **kw)
        self.blocks = [
            torch.compile(_FastBlock(b, self.num_frames, FAST_ATTN_IMPL, identity), **kw)
            for b in model.blocks
        ]
        self.cross_kv = [torch.compile(_FastCrossKV(b), **kw) for b in model.blocks]
        self.head = torch.compile(
            _FastHead(model.head, self.grid, model.patch_size, model.out_dim, guide_scale), **kw)
        self.context = None
        logger.info(f"[fast_dit] grid={self.grid} L={self.seq_len} attn={FAST_ATTN_IMPL} "
                    f"fullgraph={FAST_FULLGRAPH} blocks={len(self.blocks)} cfg_batch=2")

    def set_context(self, ctx_cond, ctx_null):
        ctx = torch.cat([ctx_cond, ctx_null], dim=0).to(torch.bfloat16)  # [2, 512, 4096]
        context = self.text(ctx)
        self.context = [kv(context) for kv in self.cross_kv]  # per block (k, v)

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
        e, e0 = self._time_embed(t)
        x = self.patch(latent.to(torch.bfloat16))
        for blk, (ck, cv) in zip(self.blocks, self.context):
            x = blk(x, e0, self.cos, self.sin, ck, cv)
        return self.head(x, e)

    @staticmethod
    def log_graph_breaks():
        from torch._dynamo.utils import counters
        breaks = dict(counters.get("graph_break", {}))
        if breaks:
            logger.warning(f"[fast_dit] graph breaks ({sum(breaks.values())}): {breaks}")
        else:
            logger.info("[fast_dit] no graph breaks")
