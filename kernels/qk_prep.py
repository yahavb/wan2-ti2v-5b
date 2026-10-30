"""NKI kernels for the DiT self-attention prologue (FAST_QK_PREP): QK-norm + RoPE + the layouts
attention_cte reads, in one pass over q / k / v.

The compiled graph for this (models/fast_dit._tp_rms_qk, _rope, _cte's permutes) costs ~1.2 ms of
the 19.7 ms SP block for the norm and RoPE alone (block_bench SP_NOQKN / SP_NOROPE probes) on
12 M elements: it runs each fp32 step as a separate pass. Here every 128-token tile of q / k / v is
loaded once, normalized, rotated and written straight into the attention layouts.

Two kernels, because QK-norm is over all heads (3072 features) and this rank holds 6 of 24:
  qk_sumsq(q, k) -> ss [Np, 2] fp32: per token, the sum of squares of this rank's q / k features
  (the caller all-reduces it over the TP group), then
  qk_apply(q, k, v, ss, wq, wk, cos, sin) -> qn [H, N, d], kn [H, d, N], vn [H, N, d]:
    qn = rope(bf16(bf16(q * rsqrt(ss_q / G + eps)) * wq)), head-major (attention_cte tp_q layout)
    kn = the same for k, transposed per head on the PE (tp_k=False layout)
    vn = v head-major
  with the rounding of the torch reference: fp32 products, bf16 after the norm, after the weight,
  and after the fp32 rotation. RoPE pairs features (2i, 2i+1) of a head with angle i (cos / sin
  [N, d/2] fp32, the same for every head).
q, k, v [Np, H*d] bf16 (Np >= N: the padded token rows are not read), wq / wk [H*d] bf16.
"""
import functools

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def _dma(dst, src):
    """dma_copy with hardware descriptor generation from the Sync engine: the default put every
    copy's descriptor generation on the GpSimd engine (DMA_DIRECT2D, ~0.86 us each), which
    serialized the kernel's ~490 copies (420 us of a 740 us kernel in the profile)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


@nki.jit
def qk_sumsq(q, k):
    Np, D = q.shape
    ss = nl.ndarray((Np, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    for t0 in range(0, Np, P):
        n = min(P, Np - t0)
        res = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(2):
            x = q if j == 0 else k
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            _dma(xt[0:n, 0:D], x[t0:t0 + n, 0:D])
            junk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=junk.ap(pattern=[[1, n], [0, D]]), op=nl.square, data=xt[0:n, 0:D],
                                   reduce_op=nl.add, reduce_res=res[0:n, j:j + 1])
        _dma(ss[t0:t0 + n, 0:2], res[0:n, 0:2])
    return ss


def _apply(src, ss, wq, wk, cos, sin, d, eps, inv_g, gp=False):
    """src: ((q, c_q), (k, c_k), (v, c_v)), each segment D = wq.shape[0] columns of its tensor
    starting at column c_* (three [Np, D] tensors at 0, or one fused [Np, 3D] qkv at 0, D, 2D)."""
    (q, cq), (k, ck), (v, cv) = src
    D = wq.shape[0]
    H = D // d
    half = d // 2
    N = cos.shape[0]
    qn = nl.ndarray((H, N, d), dtype=q.dtype, buffer=nl.shared_hbm)
    kn = nl.ndarray((H, d, N), dtype=k.dtype, buffer=nl.shared_hbm)
    vn = nl.ndarray((H, N, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for h in range(H):  # v: head-major, HBM -> HBM
        _dma(vn[h, 0:N, 0:d], v[0:N, cv + h * d:cv + (h + 1) * d])
    # the norm weights on every partition: ones[1, P]^T @ w[1, D] on the PE (exact)
    ones = nl.ndarray((1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    w_sb = []
    for w in (wq, wk):
        row = nl.ndarray((1, D), dtype=w.dtype, buffer=nl.sbuf)
        _dma(row, w.reshape((1, D)))
        t = nl.ndarray((P, D), dtype=w.dtype, buffer=nl.sbuf)
        for c0 in range(0, D, 512):
            cn = min(512, D - c0)
            ps = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ps[0:P, 0:cn], stationary=ones[0:1, 0:P], moving=row[0:1, c0:c0 + cn])
            nisa.tensor_copy(dst=t[0:P, c0:c0 + cn], src=ps[0:P, 0:cn])
        w_sb.append(t)
    eps_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=eps)
    for t0 in range(0, N, P):
        n = min(P, N - t0)
        sst = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        _dma(sst[0:n, 0:2], ss[t0:t0 + n, 0:2])
        r = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=r[0:n, 0:2], op=nl.rsqrt, data=sst[0:n, 0:2], scale=inv_g, bias=eps_t[0:n, 0:1])
        cs = nl.ndarray((P, half), dtype=cos.dtype, buffer=nl.sbuf)
        sn = nl.ndarray((P, half), dtype=sin.dtype, buffer=nl.sbuf)
        _dma(cs[0:n, 0:half], cos[t0:t0 + n, 0:half])
        _dma(sn[0:n, 0:half], sin[t0:t0 + n, 0:half])
        cb = cs.ap(pattern=[[half, n], [0, H], [1, half]], offset=0)  # broadcast over heads
        sb = sn.ap(pattern=[[half, n], [0, H], [1, half]], offset=0)
        for j in range(2):
            x, c0 = (q, cq) if j == 0 else (k, ck)
            xt = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            _dma(xt[0:n, 0:D], x[t0:t0 + n, c0:c0 + D])
            tn = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)  # bf16(x * rsqrt(..))
            nisa.activation(dst=tn[0:n, 0:D], op=nl.copy, data=xt[0:n, 0:D], scale=r[0:n, j:j + 1])
            u = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)  # bf16(.. * w)
            nisa.tensor_tensor(dst=u[0:n, 0:D], data1=tn[0:n, 0:D], data2=w_sb[j][0:n, 0:D], op=nl.multiply)
            ue = u.ap(pattern=[[D, n], [d, H], [2, half]], offset=0)
            uo = u.ap(pattern=[[D, n], [d, H], [2, half]], offset=1)
            a = nl.ndarray((P, H, half), dtype=nl.float32, buffer=nl.sbuf)
            b = nl.ndarray((P, H, half), dtype=nl.float32, buffer=nl.sbuf)
            c = nl.ndarray((P, H, half), dtype=nl.float32, buffer=nl.sbuf)
            e = nl.ndarray((P, H, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a[0:n, 0:H, 0:half], data1=ue, data2=cb, op=nl.multiply)
            eng = nisa.gpsimd_engine if gp else nisa.engine.unknown  # the sin products
            nisa.tensor_tensor(dst=b[0:n, 0:H, 0:half], data1=uo, data2=sb, op=nl.multiply, engine=eng)
            nisa.tensor_tensor(dst=c[0:n, 0:H, 0:half], data1=ue, data2=sb, op=nl.multiply, engine=eng)
            nisa.tensor_tensor(dst=e[0:n, 0:H, 0:half], data1=uo, data2=cb, op=nl.multiply)
            o = nl.ndarray((P, D), dtype=x.dtype, buffer=nl.sbuf)
            oe = o.ap(pattern=[[D, n], [d, H], [2, half]], offset=0)
            oo = o.ap(pattern=[[D, n], [d, H], [2, half]], offset=1)
            nisa.tensor_tensor(dst=oe, data1=a[0:n, 0:H, 0:half], data2=b[0:n, 0:H, 0:half], op=nl.subtract)
            nisa.tensor_tensor(dst=oo, data1=c[0:n, 0:H, 0:half], data2=e[0:n, 0:H, 0:half], op=nl.add)
            if j == 0:  # qn[h, t0 + p, :] = o[p, h*d:(h+1)*d], all heads in one DMA
                _dma(qn.ap(pattern=[[d, n], [N * d, H], [1, d]], offset=t0 * d),
                     o.ap(pattern=[[D, n], [d, H], [1, d]], offset=0))
            else:  # per head a PE transpose into one PSUM tile, one copy out, one DMA
                tp = nl.ndarray((d, H, P), dtype=x.dtype, buffer=nl.psum)
                for h in range(H):
                    nisa.nc_transpose(dst=tp[0:d, h, 0:n], data=o[0:n, h * d:(h + 1) * d])
                ts = nl.ndarray((d, H, P), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ts[0:d, 0:H, 0:n], src=tp[0:d, 0:H, 0:n], engine=nisa.scalar_engine)
                _dma(kn.ap(pattern=[[N, d], [d * N, H], [1, n]], offset=t0),
                     ts.ap(pattern=[[H * P, d], [P, H], [1, n]], offset=0))
    return qn, kn, vn


@functools.lru_cache(maxsize=None)
def qk_apply_kernel(d, eps, global_dim, gp=False, fused=False):
    """qk_apply for one head size / norm eps / global norm width (gp: two of the six RoPE
    products on the GpSimd engine; fused: q / k / v as one [Np, 3D] tensor, qkv_proj_kernel's
    output). Each configuration gets its own kernel name: NKI's persistent trace cache keys a
    kernel by name and argument shapes, not closure values."""
    inv_g = 1.0 / global_dim
    if fused:
        def kernel(qkv, ss, wq, wk, cos, sin):
            D = wq.shape[0]
            return _apply(((qkv, 0), (qkv, D), (qkv, 2 * D)), ss, wq, wk, cos, sin, d, eps, inv_g, gp)
    else:
        def kernel(q, k, v, ss, wq, wk, cos, sin):
            return _apply(((q, 0), (k, 0), (v, 0)), ss, wq, wk, cos, sin, d, eps, inv_g, gp)
    name = f"qk_apply_d{d}_g{global_dim}_e{eps:g}" + ("_gp" if gp else "") + ("_fused" if fused else "")
    kernel.__name__ = kernel.__qualname__ = name.replace(".", "p").replace("-", "m")
    return nki.jit(kernel)


@functools.lru_cache(maxsize=None)
def qkv_proj_kernel(n_heads, d):
    """FAST_QKV_CTE: the self-attention q / k / v projections as nkilib's qkv_cte on the fused
    weight, which also writes each token's sum of squares of its q and of its k features while the
    tile is in SBUF (the sums qk_sumsq would read q and k back for):
      kernel(x [1, Np, C], w [C, 3D], b [1, 3D]) -> qkv [1, Np, 3D], qss [1, Np, 1], kss [1, Np, 1]
    with D = n_heads * d (this rank's heads). The three compiler matmuls + qk_sumsq: 1.94 ms on one
    core at 7856 x 3072 @ 3072 x 2304; this: 1.56 ms, outputs and sums bitwise equal
    (scripts/validate/qkv_bench.py)."""
    def kernel(x, w, b):
        from nkilib.core.qkv.qkv_cte import qkv_cte
        from nkilib.core.utils.common_types import QKVOutputLayout
        B, S, _ = x.shape
        qss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        kss = nl.ndarray((B, S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        out = qkv_cte(input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.BSD, bias=b, d_head=d,
                      num_q_heads=n_heads, num_kv_heads=n_heads, q_squared_sum_out=qss, k_squared_sum_out=kss)
        return out, qss, kss
    kernel.__name__ = kernel.__qualname__ = f"qkv_proj_sumsq_h{n_heads}_d{d}"
    return nki.jit(kernel)
