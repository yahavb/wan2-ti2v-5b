"""NKI kernel for the end of a VAE upsample stage (VAE_SUBPIXEL + VAE_FUSE_SHORTCUT): interleave the
4 sub-pixel phase outputs of the upsample conv into the 2x frame and add the stage's DupUp3D
shortcut, in one pass over the data.

Today this is three eager steps: torch.stack of the 4 phases, a permute + reshape that interleaves
them (an element-strided transpose: 3.2 ms for 512 channels x 8 frames at 22x136 -> 44x272, ~60
GB/s), and the shortcut + add graph. Here each 128-channel x frame tile of the 4 phases is loaded
once, the four strided adds (phase + shortcut) write the interleaved rows in SBUF, and the frame
is stored with one contiguous DMA per partition.

Shortcut (DupUp3D, factor_s = 2): out[c, 2 t0 + ft, 2i + a, 2j + b] = x_rep[c f + 4 ft + 2a + b, t0,
i, j] with x_rep = x.repeat_interleave(repeats, dim=1) and f = 4 factor_t, i.e. x channel
(c f + 4 ft + 2a + b) // repeats. The three layouts of this VAE's decoder:
  stage 0, factor_t 1, repeats 4 (1024 -> 1024): x channel c                    (sel 0)
  stage 1, factor_t 2, repeats 8 (1024 -> 1024): x channel c, frame t // 2      (sel 0, t_div 2)
  stage 2, factor_t 2, repeats 4 (1024 -> 512):  x channel 2c + ft, frame t // 2 (sel 2, t_div 2)
(sel 1: x channel 2c + a, for factor_t 1 / repeats 2). In the first chunk the temporal shortcut
drops its first frame (t_off 1).

Layout (2-D views, no copies): p_k [C, T2*h*w] = phase (a, b) = divmod(k, 2) of the upsample conv,
x [Cx, T0*h*w] = the stage input; returns y [C, T2*2h*2w].
"""
import functools

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def _merge(p0, p1, p2, p3, x, T2, h, w, sel, t_div, t_off):
    C = p0.shape[0]
    hw = h * w
    ch_mul = 1 if sel == 0 else 2
    y = nl.ndarray((C, T2 * 4 * hw), dtype=p0.dtype, buffer=nl.shared_hbm)
    phases = (p0, p1, p2, p3)
    x3 = x.reshape((C, ch_mul, x.shape[1]))  # x channel ch_mul*c + s -> [c, s]
    for ci in range(C // P):
        c0 = ci * P
        for t in range(T2):
            u = t + t_off  # frame index before the first chunk's drop
            tx = u // t_div
            pt = []
            for k in range(4):
                tile = nl.ndarray((P, hw), dtype=p0.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=phases[k][c0:c0 + P, t * hw:(t + 1) * hw])
                pt.append(tile)
            subs = (0, 1) if sel == 1 else ((u % 2,) if sel == 2 else (0,))
            st = []
            for sub in subs:
                tile = nl.ndarray((P, hw), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=x3[c0:c0 + P, sub, tx * hw:(tx + 1) * hw])
                st.append(tile)
            o = nl.ndarray((P, h, 2, w, 2), dtype=p0.dtype, buffer=nl.sbuf)
            for a in range(2):
                for b in range(2):
                    nisa.tensor_tensor(dst=o[0:P, 0:h, a, 0:w, b], data1=pt[2 * a + b].reshape((P, h, w)),
                                       data2=st[a if sel == 1 else 0].reshape((P, h, w)), op=nl.add)
            nisa.dma_copy(dst=y[c0:c0 + P, t * 4 * hw:(t + 1) * 4 * hw], src=o.reshape((P, 4 * hw)))
    return y


@functools.lru_cache(maxsize=None)
def subpixel_merge(T2, h, w, sel, t_div, t_off):
    """The kernel for one shape / shortcut layout, wrapped for torch (traced once per process).
    Each configuration gets its own kernel name: NKI's persistent trace cache keys a kernel by
    name and argument shapes, not by closure values."""
    from torch_neuronx.nki_hop import wrap_nki

    def kernel(p0, p1, p2, p3, x):
        return _merge(p0, p1, p2, p3, x, T2, h, w, sel, t_div, t_off)
    kernel.__name__ = kernel.__qualname__ = f"subpixel_merge_T{T2}_h{h}_w{w}_s{sel}_d{t_div}_o{t_off}"
    return wrap_nki(nki.jit(kernel))


def merge_call(ys, x, factor_t, repeats, first_chunk):
    """ys: the 4 phase outputs [1, C, T2, h, w]; x: the stage input [1, Cx, T0, h, w] ->
    [1, C, T2, 2h, 2w] = interleave(ys) + DupUp3D(x) (factor_s 2)."""
    _, C, T2, h, w = ys[0].shape
    layouts = {(1, 4): (0, 1), (1, 2): (1, 1), (2, 8): (0, 2), (2, 4): (2, 2)}
    if (factor_t, repeats) not in layouts:
        raise NotImplementedError(f"DupUp3D factor_t={factor_t} repeats={repeats}")
    sel, t_div = layouts[(factor_t, repeats)]
    t_off = int(bool(first_chunk)) if factor_t == 2 else 0
    assert C % P == 0 and x.shape[1] == C * (1 if sel == 0 else 2), (C, x.shape)
    k = subpixel_merge(T2, h, w, sel, t_div, t_off)
    y = k(*[p.reshape(C, T2 * h * w) for p in ys], x.reshape(x.shape[1], -1))
    return y.view(1, C, T2, 2 * h, 2 * w)
