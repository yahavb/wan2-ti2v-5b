"""Patch WanModel self-attention blocks to use SP-aware attention.

Replaces the forward method of each block's self_attn with WanSelfAttentionSP.forward,
reusing the existing (TP-sharded) weights.
"""

from models import parallel_state as ps
from models.dit_attention_sp import WanSelfAttentionSP


def patch_model_for_sp(model):
    """Replace self-attention forward in all blocks with SP-aware version.

    The WanSelfAttentionSP module expects the same Q/K/V/O linear layers
    and norms as WanSelfAttention (already TP-sharded). We create a
    WanSelfAttentionSP instance and copy the weight references.
    """
    sp_degree = ps.get_world_size("attn-sp")
    if sp_degree <= 1:
        return model

    tp_degree = ps.get_world_size("attn-tp")
    # Global num_heads (before TP sharding) — WanSelfAttentionSP divides by tp internally
    num_heads_global = model.num_heads

    for block_idx, block in enumerate(model.blocks):
        old_attn = block.self_attn

        # Create SP attention with GLOBAL head count (it divides by TP internally)
        sp_attn = WanSelfAttentionSP(
            dim=old_attn.dim,
            num_heads=num_heads_global,
            eps=old_attn.eps)

        # Share weight references (no copy needed — same tensors)
        sp_attn.q = old_attn.q
        sp_attn.k = old_attn.k
        sp_attn.v = old_attn.v
        sp_attn.o = old_attn.o
        sp_attn.norm_q = old_attn.norm_q
        sp_attn.norm_k = old_attn.norm_k

        block.self_attn = sp_attn

    print(f"[SP] Patched {len(model.blocks)} blocks with WanSelfAttentionSP "
          f"(SP={sp_degree}, TP={tp_degree}, {num_heads_global//tp_degree} heads/rank)")
    return model
