"""NKI kernel for VAE_STRIPS: RMS_norm + SiLU of one rank's strip, written straight into the conv
input together with the neighbours' halo rows and the 2 cached frames, plus the next chunk's cache.

The compiled torch graph for this (wan/modules/vae2_2._strip_norm_cat_impl) runs at ~30-45 GB/s at
the 8x level in the pipeline (3.2 ms for C=256, 7.8 ms for C=512 per call, vs 1.3 ms for the
same norm without halo rows): the largest cost of the strip decode. Here every frame is streamed
through SBUF in 2048-pixel blocks with channels on the partitions; the channel sum of squares is a
matmul with a ones matrix (which also broadcasts it to every partition), so each element is read
once and written once.

Layout (2-D views made by the caller, no copies):
  x      [C, T*h*W]       the strip's T new frames, h rows of W
  gamma  [C, 1]           RMS_norm gain
  prev   [C, 2*(h+2)*W]   the 2 cached conv-input frames (halo rows included; zeros for chunk 0)
  g      [R*C, T*W]       every rank's normed edge rows (all-gather output), row r = g[r*C:(r+1)*C]
  ia, ib                  rows of g above / below this strip (-1 = image border: zeros)
Returns z [C, (T+2)*(h+2)*W] (frames: prev, then the T new ones with their halo rows) and
cache [C, 2*(h+2)*W] (the last 2 frames of z).
"""
import functools

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
BLK = 2048  # pixels per SBUF block (4 KB of bf16 per partition)
MM = 512    # matmul free size (one PSUM bank of fp32)


def _strip_norm_cat(x, gamma, prev, g, T, h, W, ia, ib):
    C = x.shape[0]
    CT = C // P
    H2 = h + 2
    HW = h * W
    F2 = H2 * W
    z = nl.ndarray((C, (T + 2) * F2), dtype=x.dtype, buffer=nl.shared_hbm)
    cache = nl.ndarray((C, 2 * F2), dtype=x.dtype, buffer=nl.shared_hbm)

    ones = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    eps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps, value=1e-8)
    zrow = nl.ndarray((P, T, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zrow, value=0.0)
    gam_b = nl.ndarray((P, CT), dtype=gamma.dtype, buffer=nl.sbuf)
    for ci in range(CT):
        nisa.dma_copy(dst=gam_b[0:P, ci:ci + 1], src=gamma[ci * P:(ci + 1) * P, 0:1])
    gam = nl.ndarray((P, CT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gam, src=gam_b)

    for ci in range(CT):
        c0 = ci * P
        z3 = z[c0:c0 + P, 0:(T + 2) * F2].reshape((P, T + 2, F2))
        c3 = cache[c0:c0 + P, 0:2 * F2].reshape((P, 2, F2))
        # the 2 cached frames -> z frames 0, 1 (and the cache when only 1 new frame)
        pv = nl.ndarray((P, 2 * F2), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pv, src=prev[c0:c0 + P, 0:2 * F2])
        nisa.dma_copy(dst=z[c0:c0 + P, 0:2 * F2], src=pv)
        if T == 1:
            nisa.dma_copy(dst=cache[c0:c0 + P, 0:F2], src=pv[0:P, F2:2 * F2])
        # halo rows of the new frames: row 0 and row h+1 of each frame
        for slot, src_row, row in ((0, ia, 0), (1, ib, h + 1)):
            if src_row >= 0:
                hr = nl.ndarray((P, T, W), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=hr, src=g[src_row * C + c0:src_row * C + c0 + P, 0:T * W].reshape((P, T, W)))
            else:
                hr = zrow
            nisa.dma_copy(dst=z3[0:P, 2:T + 2, row * W:(row + 1) * W], src=hr)
            k0 = max(0, T - 2)
            nisa.dma_copy(dst=c3[0:P, 2 - (T - k0):2, row * W:(row + 1) * W], src=hr[0:P, k0:T, 0:W])

    for t in range(T):
        for b0 in range(0, HW, BLK):
            nb = min(BLK, HW - b0)
            xs = []
            sqs = []
            for ci in range(CT):
                c0 = ci * P
                xt = nl.ndarray((P, BLK), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt[0:P, 0:nb], src=x[c0:c0 + P, t * HW + b0:t * HW + b0 + nb])
                sq = nl.ndarray((P, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=sq[0:P, 0:nb], data1=xt[0:P, 0:nb], data2=xt[0:P, 0:nb], op=nl.multiply)
                xs.append(xt)
                sqs.append(sq)
            rms = nl.ndarray((P, BLK), dtype=nl.float32, buffer=nl.sbuf)
            for m0 in range(0, nb, MM):
                nm = min(MM, nb - m0)
                ps = nl.ndarray((P, MM), dtype=nl.float32, buffer=nl.psum)
                for ci in range(CT):
                    nisa.nc_matmul(dst=ps[0:P, 0:nm], stationary=ones, moving=sqs[ci][0:P, m0:m0 + nm],
                                   accumulate=(ci > 0), is_stationary_onezero=True)
                nisa.activation(dst=rms[0:P, m0:m0 + nm], op=nl.rsqrt, data=ps[0:P, 0:nm],
                                scale=1.0 / C, bias=eps)
            for ci in range(CT):
                c0 = ci * P
                y = nl.ndarray((P, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=y[0:P, 0:nb], data1=xs[ci][0:P, 0:nb], data2=rms[0:P, 0:nb], op=nl.multiply)
                o = nl.ndarray((P, BLK), dtype=x.dtype, buffer=nl.sbuf)
                nisa.activation(dst=o[0:P, 0:nb], op=nl.silu, data=y[0:P, 0:nb], scale=gam[0:P, ci:ci + 1])
                off = (t + 2) * F2 + W + b0
                nisa.dma_copy(dst=z[c0:c0 + P, off:off + nb], src=o[0:P, 0:nb])
                if t >= T - 2:
                    coff = (t - (T - 2)) * F2 + W + b0
                    nisa.dma_copy(dst=cache[c0:c0 + P, coff:coff + nb], src=o[0:P, 0:nb])
    return z, cache


@functools.lru_cache(maxsize=None)
def strip_norm_cat(T, h, W, ia, ib):
    """The kernel for one shape and halo source pair, wrapped for torch (traced once per process)."""
    from torch_neuronx.nki_hop import wrap_nki

    def kernel(x, gamma, prev, g):
        return _strip_norm_cat(x, gamma, prev, g, T, h, W, ia, ib)
    # NKI's persistent trace cache keys a kernel by module.qualname + file + argument shapes, not
    # by closure values: without a per-config name, ranks with the same shapes but other halo
    # sources (ia, ib) got each other's traces.
    name = f"strip_norm_cat_T{T}_h{h}_W{W}_a{ia}_b{ib}".replace("-", "m")
    kernel.__name__ = kernel.__qualname__ = name
    return wrap_nki(nki.jit(kernel))
