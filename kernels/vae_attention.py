"""NKI kernel for VAE AttentionBlock — single-head spatial self-attention.

The VAE AttentionBlock does:
  RMSNorm → Conv2d_1x1(QKV) → scaled_dot_product_attention → Conv2d_1x1(proj) + residual

This kernel handles the core SDPA part. The 1x1 convs use vae_conv2d_k1.

Input:  q (1, d, seq), k (1, d, seq), v (1, seq, d)
Output: (seq, 1, d)

IMPORTANT: All seq dims must be padded to multiple of 512 by the caller.
NKI requires compile-time constant sizes in nl.ds() — no variable-size slices.

Production shapes:
  Decoder middle block: d=1024, seq=30*52=1560 → pad to 2048

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

    REQUIREMENT: seq_q and seq_k MUST be multiples of 512. Pad at call site.
    d MUST be a multiple of 128.
    """
    batch_size = q.shape[0]  # 1 for VAE
    d = q.shape[1]           # 1024 (or 512, 256)
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]

    P = nl.tile_size.pmax    # 128
    CHUNK = 512              # fixed chunk size — all dims padded to this

    num_q_grps = seqlen_q // P       # seq_q / 128
    num_v_tiles = seqlen_k // P      # seq_k / 128
    num_d_tiles = d // P             # d / 128
    num_sk_chunks = seqlen_k // CHUNK  # seq_k / 512

    # Output in HBM
    out = nl.ndarray((seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity matrix
    id_sbuf = nl.ndarray((P, P), dtype=identity.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for batch_id in range(batch_size):  # just 1 iteration for VAE

        for gi in range(num_q_grps):
            q_start = gi * P

            # ── Phase 1: QK^T = sum over d-tiles of Q_tile^T @ K_tile ──
            qk_acc = nisa.memset((P, seqlen_k), value=0.0, dtype=nl.float32)

            for dt in range(num_d_tiles):
                d_off = dt * P

                # Load Q tile slice: [P_d, P_seq] from q[batch_id, d_off:d_off+P, q_start:q_start+P]
                q_buf = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_buf, src=q[batch_id, nl.ds(d_off, P), nl.ds(q_start, P)])

                # Load K: [P_d, seq_k] — load in fixed 512 chunks (no variable sizes)
                k_buf = nl.ndarray((P, seqlen_k), dtype=k.dtype, buffer=nl.sbuf)
                for kc in range(num_sk_chunks):
                    kc_start = kc * CHUNK
                    nisa.dma_copy(dst=k_buf[:, nl.ds(kc_start, CHUNK)],
                                  src=k[batch_id, nl.ds(d_off, P), nl.ds(kc_start, CHUNK)])

                # nc_matmul: Q_buf[P,P].T @ K_buf[P,seq_k] → [P, seq_k]
                qk_psum = nisa.nc_matmul(q_buf, k_buf)
                qk_sbuf = nisa.tensor_copy(qk_psum)
                qk_acc[...] = nisa.tensor_tensor(qk_acc, qk_sbuf, nl.add)

            # Scale
            qk_scaled = nisa.tensor_scalar(qk_acc, nl.multiply, softmax_scale)

            # ── Phase 2: Softmax ──
            # Row max — reduce in fixed 512 chunks
            pmaxes = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * CHUNK
                pmaxes[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.maximum, qk_scaled[:, nl.ds(sc_start, CHUNK)], axis=1)
            row_max = nisa.tensor_reduce(nl.maximum, pmaxes, axis=1)

            # Subtract max and exp
            qk_shifted = nisa.tensor_tensor(qk_scaled, row_max, nl.subtract)
            exp_qk = nisa.activation(nl.exp, qk_shifted)

            # Row sum — reduce in fixed 512 chunks
            psums = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * CHUNK
                psums[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.add, exp_qk[:, nl.ds(sc_start, CHUNK)], axis=1)
            row_sum = nisa.tensor_reduce(nl.add, psums, axis=1)
            row_sum_recip = nisa.reciprocal(row_sum)

            # Cast exp to bf16 for matmul
            exp_bf16 = nisa.tensor_copy(exp_qk, dtype=nl.bfloat16)

            # ── Phase 3: PV matmul ──
            # attn_weights @ V: [P, seq_k] @ [seq_k, d] → [P, d]
            pv_accum = nisa.memset((P, d), value=0.0, dtype=nl.float32)

            for vi in range(num_v_tiles):
                v_start = vi * P

                # Load V tile: [P, d] from v[batch_id, v_start:v_start+P, :]
                # d is multiple of 128, so load in 512 chunks along d
                v_tile = nl.ndarray((P, d), dtype=v.dtype, buffer=nl.sbuf)
                d_chunks = d // CHUNK
                d_remainder = d % CHUNK
                for dc in range(d_chunks):
                    dc_start = dc * CHUNK
                    nisa.dma_copy(dst=v_tile[:, nl.ds(dc_start, CHUNK)],
                                  src=v[batch_id, nl.ds(v_start, P), nl.ds(dc_start, CHUNK)])
                # Handle d remainder (if d not multiple of 512, e.g. d=1024 is fine, d=256 is fine)
                # d=1024: 2 chunks of 512, no remainder
                # d=512: 1 chunk of 512, no remainder
                # d=256: 0 chunks of 512, remainder=256
                if d_remainder > 0:
                    # d_remainder must be a compile-time constant (it is, since d is from tensor shape)
                    dc_start_rem = d_chunks * CHUNK
                    nisa.dma_copy(dst=v_tile[:, nl.ds(dc_start_rem, d_remainder)],
                                  src=v[batch_id, nl.ds(v_start, P), nl.ds(dc_start_rem, d_remainder)])

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
