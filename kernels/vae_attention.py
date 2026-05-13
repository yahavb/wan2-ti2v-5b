"""NKI kernel for VAE AttentionBlock — single-head spatial self-attention.

The VAE AttentionBlock does:
  RMSNorm → Conv2d_1x1(QKV) → scaled_dot_product_attention → Conv2d_1x1(proj) + residual

This kernel handles the core SDPA part. The 1x1 convs (which are just matmuls) and
RMSNorm can use vae_conv2d_k1 or be fused into this kernel later.

For now, this kernel implements the SDPA portion:
  Input:  q (1, d, seq), k (1, d, seq), v (1, seq, d)
  Output: (seq, 1, d)

This is a simplified version of wan_cross_attn for single-head, where seq_q == seq_k.

Production shapes:
  Decoder middle block: d=1024, seq=30*52=1560 (at bottleneck resolution)
  These are small enough for single-pass softmax (no online softmax needed).

NKI API: bundled neuronxcc.nki.isa return-style.
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def vae_self_attention(q, k, v, identity, softmax_scale=None):
    """Single-head self-attention for VAE AttentionBlock.

    IO tensor layouts (matching wan_cross_attn convention):
        - q:        (1, d, seq_q)  single head, d=dim (e.g. 1024)
        - k:        (1, d, seq_k)  seq_k == seq_q for self-attention
        - v:        (1, seq_k, d)
        - identity: (128, 128)     used for transpose trick via nc_matmul
        - out:      (seq_q, 1, d)  output

    For VAE: d=1024 (bottleneck dim), seq=1560 (30×52 spatial)
    d is large (1024 = 8×128), so we tile along d for the QK matmul.
    seq is small enough to fit in SBUF (1560 < 2048).
    """
    batch_size = q.shape[0]  # 1 for VAE
    d = q.shape[1]           # 1024 (or 512, 256)
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]

    P = nl.tile_size.pmax    # 128
    assert seqlen_q % P == 0, f"seqlen_q ({seqlen_q}) must be multiple of P ({P}). Pad at call site."

    num_q_grps = seqlen_q // P
    num_v_tiles = seqlen_k // P
    num_d_tiles = d // P  # d/128 tiles for tiling the QK matmul

    # Output in HBM
    out = nl.ndarray((seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity matrix
    id_sbuf = nl.ndarray((P, P), dtype=identity.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for batch_id in range(batch_size):  # just 1 iteration for VAE

        # Load K: [d, seq_k] — load in d-tiles of P=128
        # For d=1024, seq_k=1560: need 8 tiles × [128, 1560]
        # seq_k may need padding to multiple of 512 for DMA
        # Actually, seq_k=1560 is fine for DMA in chunks

        for gi in range(num_q_grps):
            q_start = gi * P

            # ── Phase 1: QK^T = sum over d-tiles of Q_tile^T @ K_tile ──
            # Q tile: [d, P] at columns q_start..q_start+P
            # K: [d, seq_k]
            # QK^T = Q^T @ K = [P, seq_k], tiled over d
            qk_acc = nisa.memset((P, seqlen_k), value=0.0, dtype=nl.float32)

            for dt in range(num_d_tiles):
                d_off = dt * P

                # Load Q tile slice: [P_d, P_seq] from q[batch_id, d_off:d_off+P, q_start:q_start+P]
                q_buf = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_buf, src=q[batch_id, nl.ds(d_off, P), nl.ds(q_start, P)])

                # Load K slice: [P_d, seq_k] from k[batch_id, d_off:d_off+P, :]
                # Load in chunks of 512 along seq_k
                k_buf = nl.ndarray((P, seqlen_k), dtype=k.dtype, buffer=nl.sbuf)
                k_chunks = (seqlen_k + 512 - 1) // 512
                for kc in range(k_chunks):
                    kc_start = kc * 512
                    kc_size = 512
                    if kc_start + kc_size > seqlen_k:
                        kc_size = seqlen_k - kc_start
                    nisa.dma_copy(dst=k_buf[:, nl.ds(kc_start, kc_size)],
                                  src=k[batch_id, nl.ds(d_off, P), nl.ds(kc_start, kc_size)])

                # nc_matmul: Q_buf[P,P].T @ K_buf[P,seq_k] → [P, seq_k]
                qk_psum = nisa.nc_matmul(q_buf, k_buf)
                qk_sbuf = nisa.tensor_copy(qk_psum)
                qk_acc[...] = nisa.tensor_tensor(qk_acc, qk_sbuf, nl.add)

            # Scale
            qk_scaled = nisa.tensor_scalar(qk_acc, nl.multiply, softmax_scale)

            # ── Phase 2: Softmax ──
            # Row max
            # tile seq_k into chunks of 512 for reduce
            num_sk_chunks = (seqlen_k + 512 - 1) // 512
            pmaxes = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * 512
                sc_size = 512
                if sc_start + sc_size > seqlen_k:
                    sc_size = seqlen_k - sc_start
                pmaxes[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.maximum, qk_scaled[:, nl.ds(sc_start, sc_size)], axis=1)
            row_max = nisa.tensor_reduce(nl.maximum, pmaxes, axis=1)

            # Subtract max and exp
            qk_shifted = nisa.tensor_tensor(qk_scaled, row_max, nl.subtract)
            exp_qk = nisa.activation(nl.exp, qk_shifted)

            # Row sum
            psums = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * 512
                sc_size = 512
                if sc_start + sc_size > seqlen_k:
                    sc_size = seqlen_k - sc_start
                psums[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.add, exp_qk[:, nl.ds(sc_start, sc_size)], axis=1)
            row_sum = nisa.tensor_reduce(nl.add, psums, axis=1)
            row_sum_recip = nisa.reciprocal(row_sum)

            # Cast exp to bf16 for matmul
            exp_bf16 = nisa.tensor_copy(exp_qk, dtype=nl.bfloat16)

            # ── Phase 3: PV matmul ──
            # attn_weights @ V: [P, seq_k] @ [seq_k, d] → [P, d]
            # Tile over seq_k tiles (each P=128) and d tiles
            pv_accum = nisa.memset((P, d), value=0.0, dtype=nl.float32)

            for vi in range(num_v_tiles):
                v_start = vi * P

                # Load V tile: [P, d] from v[batch_id, v_start:v_start+P, :]
                v_tile = nl.ndarray((P, d), dtype=v.dtype, buffer=nl.sbuf)
                # Load in chunks of 512 along d
                d_chunks = (d + 512 - 1) // 512
                for dc in range(d_chunks):
                    dc_start = dc * 512
                    dc_size = 512
                    if dc_start + dc_size > d:
                        dc_size = d - dc_start
                    nisa.dma_copy(dst=v_tile[:, nl.ds(dc_start, dc_size)],
                                  src=v[batch_id, nl.ds(v_start, P), nl.ds(dc_start, dc_size)])

                # Extract attention weights for this V chunk: [P, P] from exp_bf16
                attn_chunk = nisa.tensor_copy(exp_bf16[:, nl.ds(v_start, P)])

                # Transpose via identity matmul trick
                attn_T_psum = nisa.nc_matmul(attn_chunk, id_sbuf)
                attn_T = nisa.tensor_copy(attn_T_psum)

                # nc_matmul: attn_T[P,P].T @ V[P,d] → [P, d]
                pv_psum = nisa.nc_matmul(attn_T, v_tile)
                pv_contrib = nisa.tensor_copy(pv_psum)

                pv_accum[...] = nisa.tensor_tensor(pv_accum, pv_contrib, nl.add)

            # ── Phase 4: Normalize and store ──
            pv_normed = nisa.tensor_tensor(pv_accum, row_sum_recip, nl.multiply)
            pv_out = nisa.tensor_copy(pv_normed, dtype=q.dtype)

            # Store: out[q_start:q_start+P, batch_id, :d]
            nisa.dma_copy(
                dst=out[nl.ds(q_start, P), batch_id, :d],
                src=pv_out
            )

    return out
