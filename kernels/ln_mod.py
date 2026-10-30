"""NKI kernels for the DiT block's LayerNorms on the token shard (FAST_LN_MOD).

The compiled graph spends ~0.95 ms of the 19.9 ms SP block on the three LayerNorms
(block_bench SP_NOLN probe), ~0.32 ms each on [1964, 3072] bf16, before the modulation even
starts. Here each 128-token tile is read once: bn_stats / bn_aggr for mean and variance (fp32),
the normalized tile on the Scalar engine, and the per-frame modulation rows selected on the PE by
the token -> frame one-hot (the same one-hot the torch code multiplies), straight from PSUM.

  ln_mod_kernel(eps)(x, ohT, ms, mt) -> h [S, C]:
      h = bf16(LN(x) * (1 + ms[f]) + mt[f]) in fp32,   f = the token's frame
    (WanAttentionBlock: norm1 + modulation rows (1, 0), norm2 + modulation rows (4, 3)): the
    upstream fp32 modulation, one rounding. (Rounding to bf16 after LN, 1 + scale and the product,
    as eager torch on bf16 tensors does, measured 0.0480 vs 0.0474 step-0 rel-RMS vs fp32 in the
    pipeline at the same speed.)
  ln_affine_kernel(eps)(x, w, b) -> bf16(LN(x) * w + b) in fp32 (norm3, nn.LayerNorm affine).
x [S, C] bf16, ohT [F, S] bf16 one-hot (column t = token t's frame; zero on pad tokens),
ms / mt [F, C] bf16 modulation rows, w / b [C] bf16. LN = (x - mean) * rsqrt(var + eps), biased
variance, no affine.

The modulation rows are tensor arguments, not closure constants: torch-neuronx and NKI key their
caches by kernel name and argument shapes, so two closures that differ only in which rows they
select can be served one compiled kernel (that is how norm2 once ran with norm1's rows). The only
closure value is eps, which is in the kernel name.
"""
import functools

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
BN = 512  # bn_stats chunk, and the PSUM tile width of the row selections


def _dma(dst, src):
    """dma_copy with hardware descriptor generation from the Sync engine (the default generated
    descriptors in software on the GpSimd engine, ~0.86 us per copy, serialized)."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _stats(xt, n, C, eps_t):
    """mean / rsqrt(var + eps) of the n rows of xt [P, C] -> mv [P, 2] fp32 (mean, rstd)."""
    nb = (C + BN - 1) // BN
    st = nl.ndarray((P, 6 * nb), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(nb):
        c0 = i * BN
        nisa.bn_stats(dst=st[0:n, 6 * i:6 * (i + 1)], data=xt[0:n, c0:min(C, c0 + BN)])
    mv = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.bn_aggr(dst=mv[0:n, 0:2], data=st[0:n, 0:6 * nb])
    nisa.activation(dst=mv[0:n, 1:2], op=nl.rsqrt, data=mv[0:n, 1:2], bias=eps_t[0:n, 0:1])
    return mv


def _eps_tile(eps):
    eps_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=eps)
    return eps_t


def _ln_rows(xt, n, C, eps_t, dtype):
    """LN of the n rows of the SBUF tile xt [P, C] (fp32 statistics and math), as dtype."""
    mv = _stats(xt, n, C, eps_t)
    nmr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # -mean * rstd
    nisa.tensor_scalar(dst=nmr[0:n, 0:1], data=mv[0:n, 0:1], op0=nl.multiply, operand0=mv[0:n, 1:2],
                       op1=nl.multiply, operand1=-1.0)
    y = nl.ndarray((P, C), dtype=dtype, buffer=nl.sbuf)
    nisa.activation(dst=y[0:n, 0:C], op=nl.copy, data=xt[0:n, 0:C], scale=mv[0:n, 1:2], bias=nmr[0:n, 0:1])
    return y


def _load(x, t0, n, C):
    t = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
    _dma(t[0:n, 0:C], x[t0:t0 + n, 0:C])
    return t


def _ln_mod(x, ohT, ms_h, mt_h, eps):
    """LN, 1 + scale and the product stay fp32, one rounding at the end (the upstream math)."""
    S, C = x.shape
    F = ohT.shape[0]
    mdt = nl.float32
    h = nl.ndarray((S, C), dtype=x.dtype, buffer=nl.shared_hbm)
    oh_sb = nl.ndarray((F, S), dtype=ohT.dtype, buffer=nl.sbuf)
    _dma(oh_sb, ohT[0:F, 0:S])
    ms = nl.ndarray((F, C), dtype=ms_h.dtype, buffer=nl.sbuf)  # scale rows, one per frame
    mt = nl.ndarray((F, C), dtype=mt_h.dtype, buffer=nl.sbuf)  # shift rows
    _dma(ms, ms_h[0:F, 0:C])
    _dma(mt, mt_h[0:F, 0:C])
    eps_t = _eps_tile(eps)
    for t0 in range(0, S, P):
        n = min(P, S - t0)
        xt = _load(x, t0, n, C)
        y = _ln_rows(xt, n, C, eps_t, mdt)  # LN(x)
        o = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        for c0 in range(0, C, BN):
            cn = min(BN, C - c0)
            psc = nl.ndarray((P, BN), dtype=nl.float32, buffer=nl.psum)  # the tokens' scale rows (exact)
            psh = nl.ndarray((P, BN), dtype=nl.float32, buffer=nl.psum)  # shift rows
            nisa.nc_matmul(dst=psc[0:n, 0:cn], stationary=oh_sb[0:F, t0:t0 + n], moving=ms[0:F, c0:c0 + cn])
            nisa.nc_matmul(dst=psh[0:n, 0:cn], stationary=oh_sb[0:F, t0:t0 + n], moving=mt[0:F, c0:c0 + cn])
            s1 = nl.ndarray((P, BN), dtype=mdt, buffer=nl.sbuf)  # 1 + scale
            nisa.activation(dst=s1[0:n, 0:cn], op=nl.copy, data=psc[0:n, 0:cn], bias=1.0)
            pr = nl.ndarray((P, BN), dtype=mdt, buffer=nl.sbuf)  # y * (1 + scale)
            nisa.tensor_tensor(dst=pr[0:n, 0:cn], data1=y[0:n, c0:c0 + cn], data2=s1[0:n, 0:cn], op=nl.multiply)
            nisa.tensor_tensor(dst=o[0:n, c0:c0 + cn], data1=pr[0:n, 0:cn], data2=psh[0:n, 0:cn], op=nl.add)
        _dma(h[t0:t0 + n, 0:C], o[0:n, 0:C])
    return h


def _bcast_rows(v, C):
    """v [C] (HBM) on every partition: [P, C] fp32, via ones[1, P]^T @ v[1, C] on the PE (exact)."""
    ones = nl.ndarray((1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    row = nl.ndarray((1, C), dtype=v.dtype, buffer=nl.sbuf)
    _dma(row, v.reshape((1, C)))
    out = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
    for c0 in range(0, C, BN):
        cn = min(BN, C - c0)
        ps = nl.ndarray((P, BN), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps[0:P, 0:cn], stationary=ones[0:1, 0:P], moving=row[0:1, c0:c0 + cn])
        nisa.tensor_copy(dst=out[0:P, c0:c0 + cn], src=ps[0:P, 0:cn])
    return out


def _ln_affine(x, w, b, eps):
    S, C = x.shape
    h = nl.ndarray((S, C), dtype=x.dtype, buffer=nl.shared_hbm)
    w_sb, b_sb = _bcast_rows(w, C), _bcast_rows(b, C)
    eps_t = _eps_tile(eps)
    for t0 in range(0, S, P):
        n = min(P, S - t0)
        xt = _load(x, t0, n, C)
        y = _ln_rows(xt, n, C, eps_t, nl.float32)  # LN(x), fp32
        yw = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=yw[0:n, 0:C], data1=y[0:n, 0:C], data2=w_sb[0:n, 0:C], op=nl.multiply)
        o = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o[0:n, 0:C], data1=yw[0:n, 0:C], data2=b_sb[0:n, 0:C], op=nl.add)
        _dma(h[t0:t0 + n, 0:C], o[0:n, 0:C])
    return h


def _named(fn, name):
    fn.__name__ = fn.__qualname__ = name.replace(".", "p").replace("-", "m")
    return nki.jit(fn)


# Build these kernels outside torch.compile'd code (e.g. in a module's __init__): dynamo ignores
# the lru_cache and traces the factory itself, re-creating the kernel inside every traced graph.
@functools.lru_cache(maxsize=None)
def ln_mod_kernel(eps):
    """Kernel for one eps, uniquely named: NKI's persistent trace cache keys a kernel by name and
    argument shapes, not by closure values. (The `_f32` suffix keeps it apart from the traces of
    an earlier bf16-step version, which was named without it.)"""
    def kernel(x, ohT, ms, mt):
        return _ln_mod(x, ohT, ms, mt, eps)
    return _named(kernel, f"ln_mod_e{eps:g}_f32")


@functools.lru_cache(maxsize=None)
def ln_affine_kernel(eps):
    def kernel(x, w, b):
        return _ln_affine(x, w, b, eps)
    return _named(kernel, f"ln_affine_e{eps:g}")
