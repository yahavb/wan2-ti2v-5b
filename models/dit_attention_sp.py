"""SP-aware self-attention for Wan2.2-TI2V-5B DiT (bidirectional diffusion).

Unlike rolling-forcing (causal, KV cache), the diffusion model uses full
bidirectional attention per denoising step. SP splits the sequence: each
rank computes attention for L/SP query tokens against ALL L key tokens.

SP=2, TP=4 layout (8 ranks total on 1 NeuronDevice, LNC1):
  - TP groups: [0,1,2,3] and [4,5,6,7]  (4 ranks share heads)
  - SP groups: [0,4], [1,5], [2,6], [3,7] (2 ranks share sequence)
  - 24 heads / TP=4 = 6 heads per rank
  - seq_len=8190 / SP=2 = 4095 Q tokens per rank

The NST kernel handles Q/K length mismatch natively via actual_seqlen_k.
"""
import math
import torch
import torch.nn as nn

from models import parallel_state as ps
from kernels.self_attention_nst import wan_flash_self_attn as wan_flash_self_attn_nst

ATTN_SEQLEN_MULTIPLE = 512


class WanSelfAttentionSP(nn.Module):
    """SP-aware bidirectional self-attention for diffusion denoising."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        assert qk_norm
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.tp_degree = ps.get_world_size("attn-tp")
        self.sp_degree = ps.get_world_size("attn-sp")
        self.tp_rank = ps.get_rank("attn-tp")
        self.sp_rank = ps.get_rank("attn-sp")
        self.heads_per_shard = num_heads // self.tp_degree

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.Module()  # placeholder, replaced by shard_model_tp
        self.norm_k = nn.Module()

        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """SP-aware bidirectional self-attention.

        Args:
            x: [1, L, dim] — full sequence (will be SP-sharded internally)
                OR [1, L/SP, dim] if pre-sharded
            seq_lens: sequence lengths [B]
            grid_sizes: [B, 3] containing (F, H, W)
            freqs: RoPE angles [max_seq, dim//2]
        """
        from wan.modules.rope_neuron import rope_apply_neuron

        b, s, _ = x.shape
        n_local = self.heads_per_shard
        d = self.head_dim
        L = int(seq_lens[0].item()) if seq_lens is not None else s

        # SP shard the sequence if input is full-length
        if s == L and self.sp_degree > 1:
            assert L % self.sp_degree == 0
            sp_shard_len = L // self.sp_degree
            sp_start = self.sp_rank * sp_shard_len
            x_local = x[:, sp_start:sp_start + sp_shard_len]
        else:
            x_local = x
            sp_shard_len = s

        # QKV on local SP shard (ColumnParallel: output is n_local * d)
        q_local = self.norm_q(self.q(x_local)).view(b, sp_shard_len, n_local, d)
        k_local = self.norm_k(self.k(x_local)).view(b, sp_shard_len, n_local, d)
        v_local = self.v(x_local).view(b, sp_shard_len, n_local, d)

        # AllGather K, V across SP to get full sequence
        if self.sp_degree > 1:
            k_full = torch.empty(b, L, n_local, d, dtype=k_local.dtype, device=k_local.device)
            v_full = torch.empty(b, L, n_local, d, dtype=v_local.dtype, device=v_local.device)
            ps.all_gather_into_tensor(
                k_full.view(L, n_local * d),
                k_local.view(sp_shard_len, n_local * d), "attn-sp")
            ps.all_gather_into_tensor(
                v_full.view(L, n_local * d),
                v_local.view(sp_shard_len, n_local * d), "attn-sp")

            # AllGather Q for correct RoPE positions
            q_full = torch.empty(b, L, n_local, d, dtype=q_local.dtype, device=q_local.device)
            ps.all_gather_into_tensor(
                q_full.view(L, n_local * d),
                q_local.view(sp_shard_len, n_local * d), "attn-sp")
        else:
            q_full = q_local
            k_full = k_local
            v_full = v_local

        # RoPE on full Q and K (correct 3D positional encoding)
        q_roped = rope_apply_neuron(q_full, grid_sizes, freqs)
        k_roped = rope_apply_neuron(k_full, grid_sizes, freqs)

        # Slice Q back to SP shard (the whole point: fewer Q tokens = less compute)
        if self.sp_degree > 1:
            sp_start_token = self.sp_rank * sp_shard_len
            q_roped_local = q_roped[:, sp_start_token:sp_start_token + sp_shard_len]
        else:
            q_roped_local = q_roped

        # Run attention via NST kernel
        # Kernel expects: q(bs, d, seq_q), k(bs, d, seq_k), v(bs, seq_k, d)
        q_kern = q_roped_local[0].permute(1, 2, 0).contiguous()  # [n_local, d, L/SP]
        k_kern = k_roped[0].permute(1, 2, 0).contiguous()        # [n_local, d, L]
        v_kern = v_full[0].permute(1, 0, 2).contiguous()          # [n_local, L, d]

        # Pad seqlen_q to multiple of 128, seqlen_k to ATTN_SEQLEN_MULTIPLE
        P = 128
        pad_q = (P - sp_shard_len % P) % P
        pad_k = (ATTN_SEQLEN_MULTIPLE - L % ATTN_SEQLEN_MULTIPLE) % ATTN_SEQLEN_MULTIPLE
        if pad_q > 0:
            q_kern = torch.nn.functional.pad(q_kern, (0, pad_q))
        if pad_k > 0:
            k_kern = torch.nn.functional.pad(k_kern, (0, pad_k))
            v_kern = torch.nn.functional.pad(v_kern, (0, 0, 0, pad_k))

        out = wan_flash_self_attn_nst(
            q_kern, k_kern, v_kern,
            softmax_scale=self.softmax_scale,
            actual_seqlen_k=L,
            use_dynamic_loop=True,
        )
        # out: [seq_q_padded, n_local, d] → slice → [1, L/SP, dim_local]
        out = out[:sp_shard_len].unsqueeze(0).flatten(2)

        # O projection (RowParallel handles all-reduce within TP group)
        out = self.o(out)

        # All-gather output across SP to return full [1, L, dim]
        if self.sp_degree > 1:
            out_full = torch.empty(b, L, self.dim, dtype=out.dtype, device=out.device)
            ps.all_gather_into_tensor(
                out_full.view(L, self.dim),
                out.view(sp_shard_len, self.dim), "attn-sp")
            out = out_full

        return out
