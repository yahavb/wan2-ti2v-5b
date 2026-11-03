# wan2-ti2v-5b: derived from nkilib's experimental attention_const_max (the version installed in the
# wan5b container), used by FAST_CMAX=1 FAST_CMAX_FLAT=1 (models/fast_dit.py). Same math and the same
# instructions per KV tile; what differs is data movement and the order of the steps, marked "wan2:".
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constant-max attention as one software pipeline over all heads and query blocks, with each
head's Q, K and V resident in SBUF.

Computes: Output = softmax(Q @ K.T * scale - const_max) @ V

  attention_cmax(q [N, d, Sq], k [N, d, Sk], v [N, Sk, d], softmax_max, softmax_scale) -> o [N, Sq, d]
  (bf16, d = 128, trn2 / LNC1; q and k head-major, as models/fast_dit's qk_apply writes them with q_t)

Per 128-key tile i of a 512-query block the work is nkilib's: MM1(i) = K_i^T Q -> scores (keys on
partitions), exp(i) = exp(scale * scores - max) on the Scalar engine -> bf16 probabilities, and
MM2(i) = P_i^T [V_i | 1] into four [128, d + 1] PSUM accumulators (the ones column gives the softmax
sums), MM2 eight tiles behind MM1. nkilib runs that pipeline once per query block, so at each of the
16 x 6 block boundaries of this call (6 heads x 7854 tokens) the Tensor engine drains the last
eight MM2s while the Scalar engine waits, then runs the next block's eight MM1s while the Scalar
engine catches up. wan2: the steps of all blocks and heads form one pipeline (the next block's
MM1 + exp run while this block's last MM2s drain), and
  - each head's Q [d, Sq], K [d, Sk] and V in the [V | 1] layout [128, Sk / 128, d + 1] are loaded
    into SBUF with one DMA each (hardware descriptors from the Sync engine), two heads ahead; nkilib
    streams K / V in 2048-key blocks for every query block (16 x 4 MB per head) and copies each V
    tile into the [V | 1] layout on the Vector engine;
  - a fresh probabilities ring and accumulator pool per query block, as nkilib allocates them per
    block call: one ring reused by all ~6000 steps made the compiler order each MM2 behind the
    newest exp (the PE and the Scalar engine then took turns, 10.3 ms).
Output bitwise equal to nkilib's at this shape; one core 3.26 -> 3.13 ms, all 8 cores of a device
running it 3.66 -> 3.51 ms per call (scripts/validate/cmax_resident_bench.py, pwr_bench.py
KERNEL=cmaxf).

The floor on each row's softmax sum is 1e-30 (nkilib: 1e-9): with a constant max m a row's sum is
about Sk * exp(typical score - m), which falls under 1e-9 once m exceeds ~30 (FAST_CMAX's shifted
maxima, CMAX_MAX_M); fp32 holds 1e-30.

wan2: check=(ln_lo, ln_hi) (the factories' argument; fast_dit's CMAX_CHECK). A constant max is
exact only while every row's exponentials stay representable, and no bound from the weights
guarantees that (the QK-norm runs over all heads, so one head's scores can reach n_heads times the
per-head weight bound). So the kernel checks its own result: every row's softmax sum S (the ones
column) must lie in [e^ln_lo, e^ln_hi]. S <= e^78 means no exponential and no partial sum
overflowed (all terms are positive); S >= e^-60 means the largest term (>= S / Sk) is a normal float
and the terms lost to underflow are below fp32 rounding. A NaN fails both comparisons. The sums are
the floored values the normalization already computes (written to a per-call buffer instead of a
scratch tile, so the output is bitwise the same): the floor (1e-30 = e^-69.1) is below the band, so
a sum of 0, or a NaN the maximum drops, still fails. The kernel then also returns stats [128, 3]
f32: per partition, the number of query rows outside the band, the smallest and the largest S
(padding rows excluded). Callers redo out-of-band calls with exact attention.
"""

import functools
from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.experimental import neurotile as nt

_P = nl.tile_size.pmax  # 128: partition dimension
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_EPS = 1e-30  # wan2: floor for sum_exp (nkilib: 1e-9)
_PROLOGUE = 8  # MM2 runs this many KV tiles behind MM1 + exp (nkilib's prologue_depth)


def _dma(dst, src):
    """wan2: hardware descriptor generation from the Sync engine for the few large loads."""
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


@nki.jit
def attention_cmax(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
) -> nl.NkiTensor:
    """softmax(Q @ K.T * softmax_scale - softmax_max) @ V; see the module docstring.

    Args:
        q_hbm: [N, d, Sq] bf16 @ HBM, query (head-major).
        k_hbm: [N, d, Sk] bf16 @ HBM, key (head-major).
        v_hbm: [N, Sk, d] bf16 @ HBM, value.
        softmax_max: constant max (scaled-score units), a Python scalar or a [128, 1] f32 HBM
            tensor; the output does not depend on it as long as exp(score - max) stays in range.
        softmax_scale: scale applied to the scores, default 1/sqrt(d).
    Returns:
        [N, Sq, d] bf16 @ HBM.
    """
    Nq, d, Sq = q_hbm.shape
    _, d_k, Sk = k_hbm.shape
    Sv, d_v = v_hbm.shape[1], v_hbm.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    _check_args(q_hbm, k_hbm, v_hbm, d, d_k, d_v, Sv, Sk)
    sm = _prepare_softmax_max(softmax_max)
    o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
    _compute(q_hbm, k_hbm, v_hbm, o_hbm, sm, softmax_scale)
    return o_hbm


def _check_args(q_hbm, k_hbm, v_hbm, d, d_k, d_v, Sv, Sk):
    kernel_assert(d == _P, f"d must be pmax (128), got {d}")
    kernel_assert(d_k == d and d_v == d, f"K / V head dims must match Q, got {d_k=}, {d_v=}, {d=}")
    kernel_assert(Sv == Sk, f"V seq length must match K, got {Sv=}, {Sk=}")
    for t in (q_hbm, k_hbm, v_hbm):
        kernel_assert(t.dtype == nl.bfloat16, f"q / k / v must be bf16, got {t.dtype}")
    kernel_assert(nl.num_programs() == 1, "attention_cmax: LNC1 only (one program, all heads)")
    kernel_assert(nisa.get_nc_version() < nisa.nc_version.gen4,
                  "attention_cmax: trn2 only (fp32 scores, Scalar-engine exp)")


@functools.lru_cache(maxsize=None)
def attention_cmax_checked(ln_lo, ln_hi):
    """attention_cmax with the row-sum check (module docstring): kernel(q, k, v, softmax_max,
    softmax_scale) -> (o [N, Sq, d] bf16, stats [128, 3] f32)."""

    def kernel(q_hbm, k_hbm, v_hbm, softmax_max, softmax_scale=None):
        Nq, d, Sq = q_hbm.shape
        _, d_k, Sk = k_hbm.shape
        Sv, d_v = v_hbm.shape[1], v_hbm.shape[2]
        softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
        _check_args(q_hbm, k_hbm, v_hbm, d, d_k, d_v, Sv, Sk)
        sm = _prepare_softmax_max(softmax_max)
        o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
        st_hbm = nl.ndarray(shape=(_P, 3), dtype=nl.float32, buffer=nl.shared_hbm)
        _compute(q_hbm, k_hbm, v_hbm, o_hbm, sm, softmax_scale, check=(ln_lo, ln_hi, st_hbm))
        return o_hbm, st_hbm

    kernel.__name__ = kernel.__qualname__ = _check_name("attention_cmax_chk", ln_lo, ln_hi)
    return nki.jit(kernel)


def _check_name(base, ln_lo, ln_hi):
    """NKI caches a traced kernel by name and argument shapes, so each band gets its own name."""
    return f"{base}_{ln_lo:g}_{ln_hi:g}".replace(".", "p").replace("-", "m")


def _prepare_softmax_max(softmax_max):
    """The Scalar-engine activation adds a bias, so pass the negated max (nkilib's convention)."""
    if isinstance(softmax_max, (int, float)):
        return -softmax_max
    max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=max_sbuf, src=softmax_max[:_P, 0:1])
    neg_max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max_sbuf, data=max_sbuf, op0=nl.multiply, operand0=-1.0)
    return neg_max_sbuf


def _buffer_sets(Nq, Sq, Sk, d):
    """Two buffer sets, each one head's Q [d, Sq], K [d, Sk] and V in the [V | 1] layout
    [128, Sk / 128, d + 1] (~47 KB per partition at 7854 tokens); head h uses set h % 2."""
    n_kv = div_ceil(Sk, _P)
    sets = []
    for _ in range(min(2, Nq)):
        q_sb = nl.ndarray(shape=(_P, Sq), dtype=nl.bfloat16, buffer=nl.sbuf)
        k_sb = nl.ndarray(shape=(_P, Sk), dtype=nl.bfloat16, buffer=nl.sbuf)
        v_sb = nl.ndarray(shape=(_P, n_kv, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(v_sb[:, :, d:d + 1], 1.0)  # the ones column, written once (the loads skip it)
        sets.append((q_sb, k_sb, v_sb))
    return sets


def _load_v(v_sb, src, rows, d):
    """V rows [rows, d] (an HBM AP-able view, row r at element offset r * row_stride) -> v_sb in the
    [V | 1] layout: key t * 128 + p on partition p, tile t. src: (tensor, row_stride, offset)."""
    t, stride, off = src
    n_kv = div_ceil(rows, _P)
    last_ext = rows - (n_kv - 1) * _P
    n_full = n_kv if last_ext == _P else n_kv - 1
    if n_full > 0:
        _dma(v_sb[:, 0:n_full, 0:d], t.ap(pattern=[[stride, _P], [_P * stride, n_full], [1, d]], offset=off))
    if n_full < n_kv:
        _dma(v_sb[0:last_ext, n_kv - 1, 0:d],
             t.ap(pattern=[[stride, last_ext], [1, d]], offset=off + n_full * _P * stride))


def _compute(q_hbm, k_hbm, v_hbm, o_hbm, softmax_max, softmax_scale, check=None):
    """q / k [N, d, S], v [N, Sk, d], o [N, Sq, d]; check: (ln_lo, ln_hi, stats [128, 3] HBM) or None."""
    Nq, d, Sq = q_hbm.shape
    Sk = k_hbm.shape[2]
    sets = _buffer_sets(Nq, Sq, Sk, d)

    def load(h):
        q_sb, k_sb, v_sb = sets[h % len(sets)]
        _dma(k_sb, k_hbm[h])
        _dma(q_sb, q_hbm[h])
        _load_v(v_sb, (v_hbm, d, h * Sk * d), Sk, d)

    nb = div_ceil(Sq, nl.tile_size.gemm_moving_fmax)

    def on_block_done(B):  # head h's buffer set is free once its last block is done (program order)
        h, b = B // nb, B % nb
        if b == nb - 1 and h + 2 < Nq:
            load(h + 2)

    load(0)
    if Nq > 1:
        load(1)
    _pipeline(Nq, Sq, Sk, d, sets, o_hbm, softmax_max, softmax_scale, on_block_done, check)


@functools.lru_cache(maxsize=None)
def attention_cmax_qk_kernel(eps, global_dim, check=None):
    """FAST_CMAX_QK: models/fast_dit's qk_apply (kernels/qk_prep: QK-norm + RoPE + the head-major
    layouts) folded into attention_cmax, so the Vector engine prepares head h + 1 while the Tensor and
    Scalar engines run head h's attention (qk_apply alone: ~0.75 ms per block, Vector-bound):
      kernel(qkv [Np, 3D], ss [Np, 2], wq [D], wk [D], cos [N, d/2], sin [N, d/2], softmax_max,
             softmax_scale) -> o [D / d, N, d]
    qkv: nkilib qkv_cte's fused output (q | k | v, this rank's D = H * d features each); ss: per
    token, the all-reduced sums of squares of q and of k over all heads; QK-norm over global_dim
    features with eps. Same per-element operations and roundings as qk_apply (fp32 products, bf16
    after the norm, after the weight and after the rotation), so the output equals qk_apply +
    attention_cmax's. Each configuration gets its own kernel name (NKI's trace cache keys a kernel by
    name and argument shapes, not closure values). check=(ln_lo, ln_hi): the row-sum check (module
    docstring); the kernel then returns (o, stats [128, 3] f32)."""
    inv_g = 1.0 / global_dim

    def kernel(qkv, ss, wq, wk, cos, sin, softmax_max, softmax_scale=None):
        Np, D3 = qkv.shape
        N, half = cos.shape
        d = 2 * half
        D = D3 // 3
        softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
        kernel_assert(d == _P, f"head dim must be pmax (128), got {d}")
        kernel_assert(D3 == 3 * D and D % d == 0 and wq.shape[0] == D, f"qkv [Np, 3D], got {qkv.shape}, D={wq.shape}")
        kernel_assert(nl.num_programs() == 1, "attention_cmax_qk: LNC1 only")
        kernel_assert(nisa.get_nc_version() < nisa.nc_version.gen4, "attention_cmax_qk: trn2 only")
        sm = _prepare_softmax_max(softmax_max)
        o_hbm = nl.ndarray(shape=(D // d, N, d), dtype=qkv.dtype, buffer=nl.shared_hbm)
        chk = None
        if check is not None:
            st_hbm = nl.ndarray(shape=(_P, 3), dtype=nl.float32, buffer=nl.shared_hbm)
            chk = (check[0], check[1], st_hbm)
        _compute_qk(qkv, ss, wq, wk, cos, sin, o_hbm, sm, softmax_scale, eps, inv_g, chk)
        return o_hbm if check is None else (o_hbm, st_hbm)

    name = f"attention_cmax_qk_g{global_dim}_e{eps:g}"
    if check is not None:
        name = _check_name(name + "_chk", *check)
    kernel.__name__ = kernel.__qualname__ = name.replace(".", "p").replace("-", "m")
    return nki.jit(kernel)


def _compute_qk(qkv, ss, wq, wk, cos, sin, o_hbm, softmax_max, softmax_scale, eps, inv_g, check=None):
    Np, D3 = qkv.shape
    N, half = cos.shape
    d = 2 * half
    D = D3 // 3
    H = D // d
    n_t = div_ceil(N, _P)  # token tiles: token t * 128 + p on partition p
    last_ext = N - (n_t - 1) * _P
    n_full = n_t if last_ext == _P else n_t - 1
    blk = nl.tile_size.gemm_moving_fmax
    nb = div_ceil(N, blk)  # query blocks per head = prep chunks per head (blk / 128 token tiles each)
    tpc = blk // _P
    sets = _buffer_sets(H, N, N, d)

    # r[p, t, j] = rsqrt(ss[t * 128 + p, j] / G + eps), j = 0 (q) / 1 (k): qk_apply's activation
    sst = nl.ndarray((_P, n_t, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sst, value=1.0)  # rows past N stay finite
    _dma(sst[:, 0:n_full, :], ss.ap(pattern=[[2, _P], [2 * _P, n_full], [1, 2]], offset=0))
    if n_full < n_t:
        _dma(sst[0:last_ext, n_t - 1, :], ss.ap(pattern=[[2, last_ext], [1, 2]], offset=n_full * _P * 2))
    eps_t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=eps)
    r_all = nl.ndarray((_P, n_t, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=r_all, op=nl.rsqrt, data=sst, scale=inv_g, bias=eps_t)
    # the norm weights on every partition
    w_sb = []
    for w in (wq, wk):
        t = nl.ndarray((_P, D), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=w.ap(pattern=[[0, _P], [1, D]], offset=0))
        w_sb.append(t)

    def segs(c):  # chunk c's token tiles as (first tile, tiles, partitions): full tiles, then the partial one
        t0, t1 = c * tpc, min(c * tpc + tpc, n_t)
        fe = min(t1, n_full)
        out = [(t0, fe - t0, _P)] if fe > t0 else []
        return out + ([(fe, 1, last_ext)] if t1 > fe else [])

    def prep(h, c):
        """Head h, token chunk c: q and k normalized and rotated, token-major [p, tile, d] bf16 (Vector)."""
        res = ([], [])
        for ts, ntl, npt in segs(c):
            cs = nl.ndarray((_P, ntl, half), dtype=cos.dtype, buffer=nl.sbuf)
            sn = nl.ndarray((_P, ntl, half), dtype=sin.dtype, buffer=nl.sbuf)
            _dma(cs[0:npt], cos.ap(pattern=[[half, npt], [_P * half, ntl], [1, half]], offset=ts * _P * half))
            _dma(sn[0:npt], sin.ap(pattern=[[half, npt], [_P * half, ntl], [1, half]], offset=ts * _P * half))
            for j in (0, 1):
                x = nl.ndarray((_P, ntl, d), dtype=qkv.dtype, buffer=nl.sbuf)
                _dma(x[0:npt], qkv.ap(pattern=[[D3, npt], [_P * D3, ntl], [1, d]], offset=ts * _P * D3 + j * D + h * d))
                tn = nl.ndarray((_P, ntl, d), dtype=qkv.dtype, buffer=nl.sbuf)  # bf16(x * rsqrt(..))
                nisa.tensor_tensor(dst=tn[0:npt], data1=x[0:npt], op=nl.multiply,
                                   data2=r_all.ap(pattern=[[n_t * 2, npt], [2, ntl], [0, d]], offset=ts * 2 + j))
                u = nl.ndarray((_P, ntl, d), dtype=qkv.dtype, buffer=nl.sbuf)  # bf16(.. * w)
                nisa.tensor_tensor(dst=u[0:npt], data1=tn[0:npt], op=nl.multiply,
                                   data2=w_sb[j].ap(pattern=[[D, npt], [0, ntl], [1, d]], offset=h * d))
                ue = u.ap(pattern=[[ntl * d, npt], [d, ntl], [2, half]], offset=0)
                uo = u.ap(pattern=[[ntl * d, npt], [d, ntl], [2, half]], offset=1)
                pa, pb, pc, pe = [nl.ndarray((_P, ntl, half), dtype=nl.float32, buffer=nl.sbuf) for _ in range(4)]
                nisa.tensor_tensor(dst=pa[0:npt], data1=ue, data2=cs[0:npt], op=nl.multiply)
                nisa.tensor_tensor(dst=pb[0:npt], data1=uo, data2=sn[0:npt], op=nl.multiply)
                nisa.tensor_tensor(dst=pc[0:npt], data1=ue, data2=sn[0:npt], op=nl.multiply)
                nisa.tensor_tensor(dst=pe[0:npt], data1=uo, data2=cs[0:npt], op=nl.multiply)
                o = nl.ndarray((_P, ntl, d), dtype=qkv.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o.ap(pattern=[[ntl * d, npt], [d, ntl], [2, half]], offset=0),
                                   data1=pa[0:npt], data2=pb[0:npt], op=nl.subtract)
                nisa.tensor_tensor(dst=o.ap(pattern=[[ntl * d, npt], [d, ntl], [2, half]], offset=1),
                                   data1=pc[0:npt], data2=pe[0:npt], op=nl.add)
                res[j].append((o, ntl, npt))
        return res

    def place(h, c, res):
        """PE transposes of prep(h, c) into one PSUM tile per tensor, one Vector copy each into the
        head's resident Q / K (columns c * blk ..)."""
        q_sb, k_sb, _ = sets[h % len(sets)]
        for j in (0, 1):
            tp = nl.ndarray((_P, blk), dtype=qkv.dtype, buffer=nl.psum)
            col = 0
            for o, ntl, npt in res[j]:
                for jj in range(ntl):
                    nisa.nc_transpose(dst=tp[:, col:col + npt], data=o[0:npt, jj, :])
                    col += npt
            nisa.tensor_copy(dst=(q_sb if j == 0 else k_sb)[:, c * blk:c * blk + col], src=tp[:, 0:col],
                             engine=nisa.engine.vector)

    def load_v(h):
        _load_v(sets[h % len(sets)][2], (qkv, D3, 2 * D + h * d), N, d)

    pending = {}

    def on_block_done(B):
        """Block b of head h done: place head h + 1's chunk b (prepared a block ago), prepare chunk
        b + 1; after head h's last block its buffer set is free: head h + 2's V, chunk 0."""
        h, b = B // nb, B % nb
        if h + 1 < H:
            place(h + 1, b, pending.pop((h + 1, b)))
            if b + 1 < nb:
                pending[(h + 1, b + 1)] = prep(h + 1, b + 1)
        if b == nb - 1 and h + 2 < H:
            load_v(h + 2)
            pending[(h + 2, 0)] = prep(h + 2, 0)

    load_v(0)
    if H > 1:
        load_v(1)
    for c in range(nb):  # head 0 before the pipeline starts
        place(0, c, prep(0, c))
    if H > 1:
        pending[(1, 0)] = prep(1, 0)
    _pipeline(H, N, N, d, sets, o_hbm, softmax_max, softmax_scale, on_block_done, check)


def _pipeline(Nq, Sq, Sk, d, sets, o_hbm, softmax_max, softmax_scale, on_block_done, check=None):
    """Every (head, query block, KV tile) step in order, MM2 _PROLOGUE steps behind MM1 + exp across
    block and head boundaries; after each block's normalize + store, on_block_done(global block).
    check: (ln_lo, ln_hi, stats [128, 3] HBM) or None (the row-sum check, module docstring)."""
    n_ot = div_ceil(Sq, _O_TILE_SIZE)  # output tiles per head
    sums = None
    if check is not None:  # wan2: the floored row sums of output tile t of head h in column t * Nq + h
        sums = nl.ndarray(shape=(_P, n_ot * Nq), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sums, value=1.0)  # padding rows of each head's last tile: inside the band
    n_kv = div_ceil(Sk, _P)
    last_ext = Sk - (n_kv - 1) * _P  # keys in the last KV tile (1..128)
    blk = nl.tile_size.gemm_moving_fmax  # 512 queries per block: one PSUM bank of fp32 scores
    nb = div_ceil(Sq, blk)
    G = Nq * nb * n_kv
    depth = min(_PROLOGUE, n_kv)
    n_prob = depth + 1  # exp(i) writes slot i % n_prob while MM2(i - depth) reads slot (i - depth) % n_prob
    scalar_max = isinstance(softmax_max, (int, float))
    o_tiles = [nt.tiles(o_hbm[h], tile_size=(_O_TILE_SIZE, d)) for h in range(Nq)]
    probs, acc = {}, {}  # per query block (global index g // n_kv): probabilities ring, accumulators

    def step(g):  # -> head, block, KV tile, queries in the block, keys in the tile
        b = (g // n_kv) % nb
        i = g % n_kv
        return g // (nb * n_kv), b, i, min(blk, Sq - b * blk), (last_ext if i == n_kv - 1 else _P)

    def mm1_exp(g):
        h, b, i, sz, ext = step(g)
        q_sb, k_sb, _ = sets[h % len(sets)]
        if i == 0:
            probs[g // n_kv] = nl.ndarray(shape=(_P, n_prob, sz), dtype=nl.bfloat16, buffer=nl.sbuf)
        # MM1(i): K_i [d, ext]^T @ Q_b [d, sz] -> scores [ext, sz]
        scores = nl.ndarray(shape=(_P, blk), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=scores[0:ext, 0:sz], stationary=k_sb[:, nl.ds(i * _P, ext)],
                       moving=q_sb[:, nl.ds(b * blk, sz)])
        # exp(i): fused evict + scale + subtract max + exp
        tsm = softmax_max if scalar_max else softmax_max[0:ext, :]
        nisa.activation(dst=probs[g // n_kv][0:ext, i % n_prob, :], data=scores[0:ext, 0:sz], op=nl.exp,
                        scale=softmax_scale, bias=tsm)

    def mm2(g):
        h, b, i, sz, ext = step(g)
        v_sb = sets[h % len(sets)][2]
        n_out = div_ceil(sz, _O_TILE_SIZE)
        if i == 0:  # O_aug [sz, d + 1]: P @ V in columns 0..d-1, the softmax sums in column d
            pool = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(sz, d + 1))
            acc[g // n_kv] = [pool[ot].data for ot in range(n_out)]
        oa = acc[g // n_kv]
        pr = probs[g // n_kv]
        for ot in range(n_out):  # MM2(i): P_i^T @ [V_i | 1] into every output tile
            st = oa[ot].shape[0]
            nisa.nc_matmul(dst=oa[ot], stationary=pr[0:ext, i % n_prob, nl.ds(ot * _O_TILE_SIZE, st)],
                           moving=v_sb[0:ext, i, :], accumulate=(i != 0))
        if i == n_kv - 1:  # the block's last MM2: normalize and store it
            for ot in range(n_out):
                t = b * (blk // _O_TILE_SIZE) + ot
                col = t * Nq + h
                sum_dst = None if sums is None else sums[0:oa[ot].shape[0], col:col + 1]
                o_tiles[h][t, 0].store(_normalize(oa[ot], sum_dst))
            on_block_done(g // n_kv)

    with nl.no_reorder():
        for g in range(G):
            mm1_exp(g)
            if g >= depth:
                mm2(g - depth)
        for g in range(G - depth, G):
            mm2(g)
    if check is not None:
        _row_sum_check(sums, Nq, n_ot, Sq - (n_ot - 1) * _O_TILE_SIZE, *check)


def _row_sum_check(sums, Nq, n_ot, last_rows, ln_lo, ln_hi, st_hbm):
    """wan2: stats[p] = (rows on partition p whose sum is outside [e^ln_lo, e^ln_hi] (NaN included),
    the smallest and the largest row sum on partition p). sums [128, n_ot * Nq]: column t * Nq + h
    holds output tile t of head h, so the heads' last (partial) tiles are the last Nq columns; their
    rows last_rows..127 are padding (1.0, counted as in band, left out of the min / max). Every
    instruction starts at partition 0."""
    import math
    NT = n_ot * Nq
    lo, hi = math.exp(ln_lo), math.exp(ln_hi)
    ok = nl.ndarray(shape=(_P, NT), dtype=nl.float32, buffer=nl.sbuf)
    ok_hi = nl.ndarray(shape=(_P, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ok, data=sums, op0=nl.greater_equal, operand0=lo)
    nisa.tensor_scalar(dst=ok_hi, data=sums, op0=nl.less_equal, operand0=hi)
    nisa.tensor_tensor(dst=ok, data1=ok, data2=ok_hi, op=nl.multiply)
    n_ok = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=n_ok, op=nl.add, data=ok, axis=1, keepdims=True)
    st = nl.ndarray(shape=(_P, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=st[:, 0:1], data=n_ok, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                       operand1=float(NT))
    full = NT if last_rows == _P else NT - Nq  # columns whose 128 rows are all real
    for j, op in ((1, nl.minimum), (2, nl.maximum)):
        if full > 0:
            nisa.tensor_reduce(dst=st[:, j:j + 1], op=op, data=sums[:, 0:full], axis=1, keepdims=True)
        if full < NT:
            last = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=last[0:last_rows], op=op, data=sums[0:last_rows, full:NT], axis=1,
                               keepdims=True)
            if full > 0:
                nisa.tensor_tensor(dst=st[0:last_rows, j:j + 1], data1=st[0:last_rows, j:j + 1],
                                   data2=last[0:last_rows], op=op)
            else:
                nisa.tensor_copy(dst=st[0:last_rows, j:j + 1], src=last[0:last_rows])
    nisa.dma_copy(dst=st_hbm, src=st)


def _normalize(o_aug_psum, sum_dst=None):
    """O = O_aug[:, :d] * (1 / max(O_aug[:, d], eps)), straight from PSUM -> [sq_tile, d] bf16 SBUF.
    sum_dst: where the floored sums go (the row-sum check's buffer), else a scratch tile."""
    sq_tile = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1
    sum_exp = nl.ndarray(shape=(sq_tile, 1), dtype=nl.float32, buffer=nl.sbuf) if sum_dst is None else sum_dst
    nisa.tensor_scalar(dst=sum_exp, data=o_aug_psum[:, d:d + 1], op0=nl.maximum, operand0=_EPS)
    inv = nl.ndarray(shape=(sq_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv, data=sum_exp)
    o_out = nl.ndarray(shape=(sq_tile, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_out, data=o_aug_psum[:, :d], op0=nl.multiply, operand0=inv)
    return o_out
