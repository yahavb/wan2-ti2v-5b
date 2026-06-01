"""Attention module with NKI kernel support for Neuron.

When running on Neuron, uses NKI flash-attention kernels from kernels/ directory.
Falls back to torch.nn.functional.scaled_dot_product_attention otherwise.
"""
import math
import warnings
import os

import torch
import torch.nn.functional as F

# Flash attention flags — disabled for Neuron
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False

# ─── NKI Kernel Loading ─────────────────────────────────────────────────────
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "1") == "1"
USE_NST_SELF_ATTN = os.environ.get("USE_NST_SELF_ATTN", "0") == "1"

_nki_cross_attn = None
_nki_self_attn = None
_nki_self_attn_nst = None
_NKI_CROSS_AVAILABLE = False
_NKI_SELF_AVAILABLE = False
_NKI_SELF_NST_AVAILABLE = False

if USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.cross_attention import wan_cross_attn as _raw_cross_attn
        _nki_cross_attn = wrap_nki(_raw_cross_attn)
        _NKI_CROSS_AVAILABLE = True
        print("[attention.py] NKI cross_attention kernel: LOADED")
    except Exception as e:
        print(f"[attention.py] NKI cross_attention kernel: FAILED ({e})")

    # NST self-attention (preferred — no mask/identity needed, handles actual_seqlen_k)
    if USE_NST_SELF_ATTN:
        try:
            from kernels.self_attention_nst import wan_flash_self_attn as _nst_self_attn
            _nki_self_attn_nst = _nst_self_attn  # already @wrap_nki decorated
            _NKI_SELF_NST_AVAILABLE = True
            print("[attention.py] NKI self_attention NST kernel: LOADED")
        except Exception as e:
            print(f"[attention.py] NKI self_attention NST kernel: FAILED ({e})")

    # Fallback: original mask-based self-attention
    if not _NKI_SELF_NST_AVAILABLE:
        try:
            from torch_neuronx.nki_hop import wrap_nki as _wrap_nki_self
            from kernels.self_attention import wan_flash_self_attn as _raw_self_attn
            _nki_self_attn = _wrap_nki_self(_raw_self_attn)
            _NKI_SELF_AVAILABLE = True
            print("[attention.py] NKI self_attention kernel: LOADED")
        except Exception as e:
            print(f"[attention.py] NKI self_attention kernel: FAILED ({e})")

# Self-attention kernel requires seqlen_k to be multiple of 8192
SELF_ATTN_SEQLEN_MULTIPLE = 8192

# Identity matrix buffer (created once, reused)
_identity_matrix = None


def _get_identity(device, dtype):
    """Get or create 128x128 identity matrix for NKI transpose trick."""
    global _identity_matrix
    if _identity_matrix is None or _identity_matrix.device != device:
        _identity_matrix = torch.eye(128, dtype=dtype, device=device)
    return _identity_matrix


def _nki_cross_attention(q, k, v, dtype=torch.bfloat16):
    """Run cross-attention using NKI kernel.
    
    Input shapes: q [B, L1, n, d], k [B, L2, n, d], v [B, L2, n, d]
    Output shape: [B, L1, n, d]
    """
    b, l1, n, d = q.shape
    l2 = k.shape[1]
    
    # Kernel expects: q(bs, d, seq_q), k(bs, d, seq_k), v(bs, seq_k, d)
    # where bs = num_heads (we process B=1, heads as batch)
    assert b == 1, "NKI kernels only support batch_size=1"
    
    q_nki = q[0].permute(1, 2, 0).contiguous()   # [n, d, L1]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [n, d, L2]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [n, L2, d]
    
    # Pad seqlen_q to multiple of 128
    P = 128
    pad_q = (P - l1 % P) % P
    if pad_q > 0:
        q_nki = F.pad(q_nki, (0, pad_q))
    
    identity = _get_identity(q.device, dtype)
    softmax_scale = 1.0 / math.sqrt(d)
    
    # Call NKI kernel
    out_nki = _nki_cross_attn(q_nki, k_nki, v_nki, identity, softmax_scale=softmax_scale)
    
    # Output: [seqlen_q_padded, n, d] → slice → [1, L1, n, d]
    out = out_nki[:l1].unsqueeze(0)
    return out


def _nki_self_attention_nst(q, k, v, dtype=torch.bfloat16):
    """Run self-attention using NST kernel (no mask/identity needed).

    Input shapes: q [B, L, n, d], k [B, L, n, d], v [B, L, n, d]
    Output shape: [B, L, n, d]
    """
    b, l, n, d = q.shape
    assert b == 1, "NKI kernels only support batch_size=1"

    q_nki = q[0].permute(1, 2, 0).contiguous()   # [n, d, L]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [n, d, L]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [n, L, d]

    P = 128
    SEQLEN_MULT = 512  # NST kernel uses 512 alignment for K
    pad_q = (P - l % P) % P
    pad_k = (SEQLEN_MULT - l % SEQLEN_MULT) % SEQLEN_MULT
    if pad_q > 0:
        q_nki = F.pad(q_nki, (0, pad_q))
    if pad_k > 0:
        k_nki = F.pad(k_nki, (0, pad_k))
        v_nki = F.pad(v_nki, (0, 0, 0, pad_k))

    softmax_scale = 1.0 / math.sqrt(d)
    out_nki = _nki_self_attn_nst(
        q_nki, k_nki, v_nki,
        softmax_scale=softmax_scale,
        actual_seqlen_k=l,
        use_dynamic_loop=True)

    out = out_nki[:l].unsqueeze(0)
    return out


def _nki_self_attention(q, k, v, dtype=torch.bfloat16):
    """Run self-attention using original mask-based NKI kernel.

    Input shapes: q [B, L, n, d], k [B, L, n, d], v [B, L, n, d]
    Output shape: [B, L, n, d]
    """
    b, l, n, d = q.shape

    assert b == 1, "NKI kernels only support batch_size=1"

    q_nki = q[0].permute(1, 2, 0).contiguous()   # [n, d, L]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [n, d, L]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [n, L, d]

    # Pad seq to multiple of 128 for Q
    P = 128
    pad_q = (P - l % P) % P
    if pad_q > 0:
        q_nki = F.pad(q_nki, (0, pad_q))

    # Pad seq_k to multiple of SELF_ATTN_SEQLEN_MULTIPLE (8192)
    pad_k = (SELF_ATTN_SEQLEN_MULTIPLE - l % SELF_ATTN_SEQLEN_MULTIPLE) % SELF_ATTN_SEQLEN_MULTIPLE
    if pad_k > 0:
        k_nki = F.pad(k_nki, (0, pad_k))
        v_nki = F.pad(v_nki, (0, 0, 0, pad_k))

    seqlen_k_padded = k_nki.shape[2]
    num_sections = seqlen_k_padded // SELF_ATTN_SEQLEN_MULTIPLE

    # Build mask: (128, seqlen_k_padded) — 0 for valid, -inf for padded
    mask = torch.zeros(P, seqlen_k_padded, dtype=dtype, device=q.device)
    if pad_k > 0:
        mask[:, l:] = float('-inf')

    identity = _get_identity(q.device, dtype)
    softmax_scale = 1.0 / math.sqrt(d)

    # Call NKI kernel
    out_nki = _nki_self_attn(
        q_nki, k_nki, v_nki, identity, mask,
        softmax_scale=softmax_scale,
        num_sections=num_sections)

    # Output: [seqlen_q_padded, n, d] → slice → [1, L, n, d]
    out = out_nki[:l].unsqueeze(0)
    return out


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    is_cross_attn=False,
):
    """Unified attention function with NKI kernel support.
    
    Args:
        q, k, v: [B, seq, num_heads, head_dim]
        is_cross_attn: If True, use cross-attention kernel (small seq_k).
                       If False, use self-attention kernel (large seq_k).
    """
    # Try NKI kernels first (Neuron device)
    if q.device.type == "neuron":
        if is_cross_attn and _NKI_CROSS_AVAILABLE:
            return _nki_cross_attention(q, k, v, dtype=dtype)
        elif not is_cross_attn and _NKI_SELF_NST_AVAILABLE:
            return _nki_self_attention_nst(q, k, v, dtype=dtype)
        elif not is_cross_attn and _NKI_SELF_AVAILABLE:
            return _nki_self_attention(q, k, v, dtype=dtype)
    
    # Fallback: scaled_dot_product_attention
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. '
            'It can have a significant impact on performance.'
        )
    
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p)
    
    out = out.transpose(1, 2).contiguous()
    return out

# Alias for backward compatibility (wan/modules/__init__.py imports flash_attention)
flash_attention = attention
