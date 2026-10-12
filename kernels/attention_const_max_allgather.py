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

"""Vendored from odyssey-trainium bringup/neuron_dit/attention_const_max_allgather.py
(branch yahavb/k8s-const-max-gen @ f30fe7a: raw-unit clip fix, activation-exp switch), plus
the sum_exp floor fix 1e-12 -> 1e-36 from 7ba5df4. kernels/attention_const_max.py adds a
local (no all-gather) entry point for TP-only use.

All-gather constant-max attention: Output = softmax(Q @ K.T * scale - const_max) @ V.

Accepting a pre-known constant max lets this kernel skip the online-softmax max reduction and
the MM1 (Q @ K.T) -> MM2 (P @ V) DMA transpose, yielding a simpler pipeline.
"""

import os
from typing import Callable, Optional, Union

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.experimental import neurotile as nt

# Tiling constants
_P = nl.tile_size.pmax  # 128: partition dimension
_SK_BLOCK = 2048  # KV streaming block size (16 tiles of 128)
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_KV_TILES_PER_BLOCK = _SK_BLOCK // _P  # 16: number of K/V tiles per KV block
_EPS = 1e-36  # sum_exp floor; a row whose max sits ~77 below c sums to ~e^-77, so the floor must stay under that
# NEURON_CMAG_ACT_EXP=1 forces the activation (Scalar-engine) exp path even when the
# compile target reports gen4 sub1 (B1). For trn3 A0 parts, where the target can claim
# vector-engine exp that the silicon does not run correctly. Read once at import.
_FORCE_ACT_EXP = os.environ.get("NEURON_CMAG_ACT_EXP", "0") == "1"
_SCORES_CLIP = 70.0
_MM1_BUFFER_BANKS = 2  # PSUM banks reserved for MM1 scores
_MAX_O_AUG_BANKS_CAP = 8  # cap on PSUM banks used for O_aug accumulation


@nki.jit
def attention_const_max_allgather(
    q_hbm: nl.NkiTensor,
    k_chunks: list,
    v_chunks: list,
    k_last: nl.NkiTensor,
    v_last: nl.NkiTensor,
    k_last_gathered: nl.NkiTensor,
    v_last_gathered: nl.NkiTensor,
    replica_group: ReplicaGroup,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
) -> nl.NkiTensor:
    """Constant-max attention over gathered + local K/V: softmax(Q @K.T * scale - max) @ V.

    k_chunks / v_chunks are already-gathered (full) K/V chunks. K chunks are supplied pre-transposed
    as [N, d, Sk_i] so their MM1 loads skip the transpose; V chunks are [N, Sk_i, d]. k_last / v_last
    are this rank's shard of one more chunk: at the start of the kernel a per-head all-gather
    reconstructs that chunk across ranks. V is gathered straight into v_last_gathered ([N, Sk_i, d]);
    K is gathered into an internal tmp [N, Sk_i, d] buffer, transpose-loaded for MM1, and its
    transposed tiles are written into k_last_gathered ([N, d, Sk_i]) — so the gathered K output is
    materialized pre-transposed without a separate transpose pass. The reconstructed chunk is
    appended to the chunk list and attention is computed over all chunks assuming a pre-known
    softmax max.

    Dimensions:
        N: Number of attention heads (batch of heads)
        d: Head dimension (must be pmax=128)
        Sq: Query sequence length
        Sk: Total Key/Value sequence length across all chunks
        Sk_i: Per-chunk Key/Value sequence length (arbitrary, including the all-gathered last chunk)
        Sk_local: This rank's Key/Value shard length within the last chunk (before all-gather)

    Args:
        q_hbm (nl.NkiTensor): [N, d, Sq] bf16 @ HBM, Query (replicated across ranks)
        k_chunks (list): Already-gathered Key chunks, each [N, d, Sk_i] bf16 @ HBM (pre-transposed
            so the MM1 load needs no transpose)
        v_chunks (list): Already-gathered Value chunks, each [N, Sk_i, d] bf16 @ HBM (aligned with k_chunks)
        k_last (nl.NkiTensor): [N, Sk_local, d] bf16 @ HBM, this rank's Key shard
        v_last (nl.NkiTensor): [N, Sk_local, d] bf16 @ HBM, this rank's Value shard
        k_last_gathered (nl.NkiTensor): [N, d, Sk_i] bf16 @ HBM, output buffer the kernel writes
            the all-gathered k_last into, pre-transposed to [N, d, Sk_i].
        v_last_gathered (nl.NkiTensor): [N, Sk_i, d] bf16 @ HBM, output buffer the
            kernel writes the all-gathered v_last into.
        replica_group (ReplicaGroup): Ranks participating in the k_last / v_last all-gather.
        softmax_max (Union[float, nl.NkiTensor]): Upper bound on (scores * scale). Output is
            mathematically invariant to this value; it only affects numerical precision. A
            reasonable estimate suffices. May be either:
              - a Python scalar (compile-time constant, broadcast to all partitions), or
              - a [128, 1] f32 HBM tensor holding a single broadcast max replicated across
                the 128 partitions. Use the tensor form to supply a data-dependent max
                computed at runtime rather than baked in at trace time.
        softmax_scale (Optional[float]): Scaling factor applied to Q. Defaults to 1/sqrt(d).

    Returns:
        output (nl.NkiTensor): [N, Sq, d] bf16 @ HBM, Normalized attention output

    Notes:
        - Requires std(Q) ≤ 1 and std(K) ≤ 1 for bf16 accuracy with softmax_max=4–5.
          This holds when QK-RMSNorm is applied after projection.
        - A softmax_max that is too low causes exp() overflow (inf/NaN in output).
          A softmax_max that is too high clips all scores toward zero, losing precision.
        - For inputs with significantly larger std use a standard online-softmax kernel.

    Choosing softmax_max from QK-RMSNorm weights:
        When QK-RMSNorm is applied, the max can be derived from the gamma weights:

            σ = sqrt(sum(γ_q² * γ_k²) / d)
            softmax_max ≈ 0.88 * σ * sqrt(2 * ln(Sk))     # Gumbel expected row-max

        As a safety check, compute the hard bound (Cauchy-Schwarz worst case):

            hard_bound = sqrt(d) * max|γ_q| * max|γ_k|

        If hard_bound - softmax_max > ~80, the score range is too wide for constant-max
        and the block should fall back to online-softmax attention.

    Pseudocode:
        all_gather(k_last -> k_last_tmp, v_last -> v_last_gathered)   # reconstruct the last chunk
        chunks = gathered_chunks + [k_last_tmp, v_last_gathered]

        for each head, each Q-block:
            Q[head, block] = load(q_hbm[head, Q-block]) * softmax_scale   # resident in SBUF

        for each chunk:
            for each head:
                for each Q-block:
                    for each kv tile:
                        K_tile = load(chunk_K_tile)  # last chunk: transpose-load + store to k_last_gathered
                        S = Q @ K_tile.T             # MM1: scores
                        P = exp(S - max)             # fused eviction + exp
                        O_aug = P @ [V_tile | 1]     # MM2: output + sum_exp
                    O_aug_acc[head, block] += O_aug  # accumulate across chunks in SBUF

        for each head, each O-tile:
            O = O_aug_acc[:, :d] / O_aug_acc[:, d]   # normalize
            store(O)
    """
    Nq, d, Sq = q_hbm.shape

    Sk_last = k_last_gathered.shape[2]
    k_last_tmp = nl.ndarray(shape=(Nq, Sk_last, d), dtype=k_last.dtype, buffer=nl.shared_hbm)
    k_chunks = k_chunks + [k_last_tmp]
    v_chunks = v_chunks + [v_last_gathered]

    d_v = v_chunks[0].shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale

    kernel_assert(d == _P, f"d must be pmax (128), got {d}")
    kernel_assert(d_v == d, f"V head dim must match Q, got {d_v=}, {d=}")
    kernel_assert(len(k_chunks) == len(v_chunks), f"K/V chunk count mismatch: {len(k_chunks)} vs {len(v_chunks)}")
    for chunk_idx in range(len(k_chunks)):
        is_last_chunk = chunk_idx == len(k_chunks) - 1
        if is_last_chunk:
            Sk_i, d_k = k_chunks[chunk_idx].shape[1], k_chunks[chunk_idx].shape[2]
        else:
            d_k, Sk_i = k_chunks[chunk_idx].shape[1], k_chunks[chunk_idx].shape[2]
        Sv_i = v_chunks[chunk_idx].shape[1]
        kernel_assert(d_k == d, f"K chunk {chunk_idx} head dim must match Q, got {d_k=}, {d=}")
        kernel_assert(Sk_i >= 1, f"K chunk {chunk_idx} Sk must be at least 1, got {Sk_i}")
        kernel_assert(Sv_i == Sk_i, f"V chunk {chunk_idx} seq length must match K, got {Sv_i=}, {Sk_i=}")
    kernel_assert(Sq >= 1, f"Sq must be at least 1, got {Sq}")

    # Prepare the constant max for the exp path once, up front. A scalar passes through as a
    # (pre-signed) float; a runtime HBM tensor is loaded into SBUF. See _prepare_softmax_max.
    prepared_softmax_max = _prepare_softmax_max(softmax_max)

    o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Shard the independent (N, Sq) work across LNC cores; K/V chunks stay replicated
    q_local, o_local, k_chunks_local, v_chunks_local, k_last_gathered_local = _shard_inputs(
        q_hbm, k_chunks, v_chunks, o_hbm, k_last_gathered
    )

    k_streams_by_head, v_streams_by_head = _build_kv_streams(k_chunks_local, v_chunks_local, k_last_gathered_local)

    # Whole tensors, never core-local slices: every core must issue the same collectives.
    collective_callbacks_by_head = _build_collective_callbacks(
        k_last, k_last_tmp, v_last, v_last_gathered, replica_group
    )

    _attention_const_max_compute(
        q_local,
        k_streams_by_head,
        v_streams_by_head,
        o_local,
        softmax_max=prepared_softmax_max,
        softmax_scale=softmax_scale,
        collective_callbacks_by_head=collective_callbacks_by_head,
    )

    return o_hbm


def _build_collective_callbacks(
    k_shard: nl.NkiTensor,
    k_out: nl.NkiTensor,
    v_shard: nl.NkiTensor,
    v_out: nl.NkiTensor,
    replica_group: ReplicaGroup,
) -> list:
    """Build per-local-head callbacks, each gathering that local index on every core.

    Callback ``l`` gathers ``{l, l + heads_per_core, ...}`` -- the same set on every core, so the
    collective sequence stays uniform (the runtime matches by position) while covering every head.
    Gathering the local index alone would miss the upper heads.

    Returns:
        list: callbacks indexed by local compute-head index.
    """
    Nq = k_shard.shape[0]
    num_cores = nl.num_programs()

    if Nq % num_cores == 0:
        # Group the global heads sharing a local index.
        heads_per_core = Nq // num_cores
        head_groups = [
            [local + core * heads_per_core for core in range(num_cores)]
            for local in range(heads_per_core)
        ]
    else:
        # Sq-sharded: local index is the global head.
        head_groups = [[head_idx] for head_idx in range(Nq)]

    def _make(heads: list):
        def _run() -> None:
            for head_idx in heads:
                ncc.all_gather(
                    dsts=[k_out[head_idx]], srcs=[k_shard[head_idx]], replica_group=replica_group,
                    collective_dim=0, priority=1
                )
                ncc.all_gather(
                    dsts=[v_out[head_idx]], srcs=[v_shard[head_idx]], replica_group=replica_group,
                    collective_dim=0, priority=1
                )

        return _run

    return [_make(group) for group in head_groups]


def _prepare_softmax_max(softmax_max: Union[float, nl.NkiTensor]) -> Union[float, nl.NkiTensor]:
    """Prepare the constant softmax max for the exp path, pre-signing it once.

    The exp path needs a specific sign convention:
      - vector_engine exponential subtracts ``max_value`` → needs positive max.
      - activation adds ``bias`` → needs negated max, so exp(scores * scale + (-max)).

    Returns:
        float if softmax_max is a scalar, else a [_P, 1] f32 SBUF tensor.
    """
    if isinstance(softmax_max, (int, float)):
        return softmax_max if _has_vector_engine_exp() else -softmax_max

    max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=max_sbuf, src=softmax_max[:_P, 0:1])
    if _has_vector_engine_exp():
        return max_sbuf

    neg_max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max_sbuf, data=max_sbuf, op0=nl.multiply, operand0=-1.0)
    return neg_max_sbuf


def _shard_inputs(
    q_hbm: nl.NkiTensor,
    k_chunks: list,
    v_chunks: list,
    o_hbm: nl.NkiTensor,
    k_last_gathered: nl.NkiTensor,
) -> tuple:
    """Shard the (N, Sq) compute across LNC cores: on heads when they divide evenly, else on Sq.

    Head sharding also slices K/V and the transpose-store destination, so each core streams only
    its own heads; Sq sharding leaves K/V whole. Uneven head counts fall back to Sq.

    Returns:
        (q_local, o_local, k_chunks_local, v_chunks_local, k_last_gathered_local).
    """
    Nq, _, Sq = q_hbm.shape
    num_cores = nl.num_programs()
    core_id = nl.program_id(0)

    if Nq % num_cores == 0:
        # Contiguous head slice per core; the callbacks mirror this grouping.
        heads_per_core = Nq // num_cores
        head_start = core_id * heads_per_core
        head_end = head_start + heads_per_core
        q_local = q_hbm[head_start:head_end, :, :]
        o_local = o_hbm[head_start:head_end, :, :]
        k_chunks_local = [k[head_start:head_end, :, :] for k in k_chunks]
        v_chunks_local = [v[head_start:head_end, :, :] for v in v_chunks]
        k_last_gathered_local = k_last_gathered[head_start:head_end, :, :]
        return q_local, o_local, k_chunks_local, v_chunks_local, k_last_gathered_local

    kernel_assert(Sq >= num_cores, f"Sq ({Sq}) must be at least num_cores ({num_cores}) to shard on Sq")

    sq_per_core = Sq // num_cores
    sq_start = core_id * sq_per_core
    # Last core takes the remainder
    sq_end = Sq if (core_id == num_cores - 1) else (sq_start + sq_per_core)

    q_local = q_hbm[:, :, sq_start:sq_end]
    o_local = o_hbm[:, sq_start:sq_end, :]
    k_chunks_local = k_chunks
    v_chunks_local = v_chunks
    k_last_gathered_local = k_last_gathered

    return q_local, o_local, k_chunks_local, v_chunks_local, k_last_gathered_local


class _StreamWrapper:
    """hardware-DGE / sync-engine loads over a `_KVBlockStream`, delegating its geometry.

    `attention_const_max_core` reads ``num_blocks`` / ``num_tiles`` / ``last_tile_extent`` off
    the stream, so we surface the wrapped stream's values here.
    """

    def __init__(self, stream: "_KVBlockStream"):
        self._stream = stream
        self.num_blocks = stream.num_blocks
        self.num_tiles = stream.num_tiles
        self.last_tile_extent = stream.last_tile_extent

    def load(self, index: int):
        return self._stream.load(index, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


class _TransposeStoreStreamWrapper:
    """Transpose-load K stream (a transposing `_KVBlockStream`) that also writes each
    transposed block back to HBM, materializing the gathered K pre-transposed."""

    def __init__(self, stream: "_KVBlockStream", k_out_head: nl.NkiTensor):
        self._stream = stream
        self.num_blocks = stream.num_blocks
        self.num_tiles = stream.num_tiles
        self.last_tile_extent = stream.last_tile_extent
        d, seq_len = k_out_head.shape
        tail_extent = seq_len % _P
        body_len = seq_len - tail_extent
        # Mirror _KVBlockStream's body/tail split so each output block matches the extent
        # of the transposed source block written into it (the tail block is sub-_P wide).
        self._body_block_count = self.num_blocks - (1 if tail_extent != 0 else 0)
        self._body_blocks = (
            nt.blocks(k_out_head[:, :body_len], tile_size=(d, _P), block_size=(1, _KV_TILES_PER_BLOCK))
            if body_len > 0
            else None
        )
        self._tail_block = (
            nt.blocks(k_out_head[:, body_len:], tile_size=(d, _P), block_size=(1, 1)) if tail_extent != 0 else None
        )
        self._store_enabled = True

    def set_store_enabled(self, enabled: bool) -> None:
        self._store_enabled = enabled

    def load(self, index: int):
        # Transpose path lowers to nisa.dma_transpose, which takes no engine/dge_mode — pass
        # neither on the load. The write-back store still uses the sync engine via hardware DGE.
        loaded = self._stream.load(index)
        if self._store_enabled:
            dst = self._body_blocks[0, index] if index < self._body_block_count else self._tail_block[0, 0]
            dst.store(loaded[0].data, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        return loaded


def _build_kv_streams(k_chunks: list, v_chunks: list, k_last_gathered: nl.NkiTensor) -> tuple:
    """Build the per-head K and V streams for all chunks, resolving transpose / gathered-K
    write-back here so the compute loop stays layout-agnostic.

    All K chunks are pre-transposed ([N, d, Sk_i]) and load directly, except the last (all-gathered
    tmp) chunk, which is [N, Sk_i, d]: it is transposed on load and its transposed blocks are also
    written back to k_last_gathered ([N, d, Sk_i]). V chunks ([N, Sk_i, d]) all load directly.

    Returns:
        (k_streams_by_head, v_streams_by_head) — each a list (per head) of per-chunk streams.
    """
    Nq = k_chunks[0].shape[0]
    num_chunks = len(k_chunks)

    k_streams_by_head = []
    v_streams_by_head = []
    for head_idx in range(Nq):
        k_streams = []
        v_streams = []
        for chunk_idx in range(num_chunks):
            is_last_chunk = chunk_idx == num_chunks - 1
            k_chunk_head = k_chunks[chunk_idx][head_idx]
            if is_last_chunk:
                # Gathered chunk arrives as [Sk_i, d]: transpose-load to [d, Sk_i] tiles and
                # write them back to k_last_gathered.
                k_kv_stream = _KVBlockStream(
                    k_chunk_head, block_size=(_KV_TILES_PER_BLOCK, 1), tiling_dim=0, transpose=True
                )
                k_stream = _TransposeStoreStreamWrapper(k_kv_stream, k_last_gathered[head_idx])
            else:
                # Pre-transposed chunk [d, Sk_i]: stream tiles directly along the Sk axis.
                k_kv_stream = _KVBlockStream(
                    k_chunk_head, block_size=(1, _KV_TILES_PER_BLOCK), tiling_dim=1, transpose=False
                )
                k_stream = _StreamWrapper(k_kv_stream)
            k_streams.append(k_stream)

            v_kv_stream = _KVBlockStream(
                v_chunks[chunk_idx][head_idx], block_size=(_KV_TILES_PER_BLOCK, 1), tiling_dim=0
            )
            v_streams.append(_StreamWrapper(v_kv_stream))
        k_streams_by_head.append(k_streams)
        v_streams_by_head.append(v_streams)

    return k_streams_by_head, v_streams_by_head


def _attention_const_max_compute(
    q_hbm: nl.NkiTensor,
    k_streams_by_head: list,
    v_streams_by_head: list,
    o_hbm: nl.NkiTensor,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: float,
    collective_callbacks_by_head: Optional[list] = None,
) -> None:
    """Core attention computation — agnostic to LNC sharding.

    Operates on the local slice of Q/O assigned to this core. K/V are supplied as pre-built
    per-head stream lists.

    Args:
        q_hbm: [N_local, d, Sq_local] bf16 @ HBM
        k_streams_by_head: list (per head) of per-chunk K streams (each yields [d, Sk_tile] tiles)
        v_streams_by_head: list (per head) of per-chunk V streams (each yields [Sk_tile, d] tiles)
        o_hbm: [N_local, Sq_local, d] bf16 @ HBM (output)
        softmax_max: pre-signed max for the exp path (float or [_P, 1] SBUF tensor)
        softmax_scale: scaling factor for Q
        collective_callbacks_by_head: optional per-head callbacks; each head's callback is fired
            once, on the first chunk's first Q-block, so its collective overlaps compute.
    """
    Nq, d, Sq = q_hbm.shape
    num_chunks = len(k_streams_by_head[0])

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    _MAX_O_AUG_BANKS = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, _MAX_O_AUG_BANKS_CAP)

    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block_size = (
        _MAX_O_AUG_BANKS * nl.tile_size.gemm_stationary_fmax if has_bf16_psum else nl.tile_size.gemm_moving_fmax
    )
    o_tiles_per_block = max_sq_block_size // _O_TILE_SIZE
    num_o_tiles_per_head = div_ceil(Sq, _O_TILE_SIZE)

    # The core loads the KV block before firing the callback, so when the gathered chunk is
    # also chunk 0 the transpose-load would read k_last_tmp before the all_gather writes it.
    # Hoist instead. O2 hits this at lead_count=0, the first chunk of a fresh KV window.
    hoist_collectives = num_chunks == 1 and collective_callbacks_by_head is not None
    if hoist_collectives:
        for head_idx in range(Nq):
            collective_callbacks_by_head[head_idx]()

    # Pre-load and scale every Q block for every head once; they stay resident in SBUF and
    # are reused across all KV chunks (O_aug accumulates over chunks). On the vector-engine
    # exp path softmax_scale is fused into the load; otherwise it is fused into the exp.
    q_blocks_by_head = []
    for head_idx in range(Nq):
        q_loader = _QBlockLoaderHeadMajor(
            q_hbm[head_idx],
            max_sq_block_size=max_sq_block_size,
            softmax_scale=softmax_scale if _has_vector_engine_exp() else None,
        )
        q_blocks_by_head.append([q_loader.load(block_idx) for block_idx in range(q_loader.num_blocks)])

    num_q_blocks = len(q_blocks_by_head[0])
    o_aug_acc = nl.ndarray(shape=(_O_TILE_SIZE, Nq, num_o_tiles_per_head, d + 1), dtype=nl.float32, buffer=nl.sbuf)

    last_chunk_idx = num_chunks - 1
    for chunk_idx in range(num_chunks):
        for head_idx in range(Nq):
            for q_block_idx in range(num_q_blocks):
                # The last chunk's stream writes its transposed tiles back to k_last_gathered; that
                # output is Q-block-independent, so only store on the first Q-block to skip redundant DMA.
                if chunk_idx == last_chunk_idx:
                    k_streams_by_head[head_idx][chunk_idx].set_store_enabled(q_block_idx == 0)
                q_block, sq_block_size = q_blocks_by_head[head_idx][q_block_idx]

                # Allocate the per-chunk output accumulator in PSUM — one tile per Sq output
                # slice, each on its own bank.
                num_output_tiles = div_ceil(sq_block_size, _O_TILE_SIZE)
                kernel_assert(
                    num_output_tiles <= _MAX_O_AUG_BANKS,
                    f"Q block too large: {num_output_tiles} O_aug tiles exceed {_MAX_O_AUG_BANKS} available PSUM banks.",
                )

                # Output accumulator width is d+1: columns 0..d-1 hold the weighted output
                # (P @ V), and column d accumulates sum_exp (P @ 1) for normalization.
                o_aug_psums = nt.psum_pool(
                    tile_size=(_O_TILE_SIZE, d + 1),
                    element_shape=(sq_block_size, d + 1),
                )

                # Fire this head's collective once, on its first chunk / first Q-block, so it
                # overlaps compute and completes before the last (gathered) chunk is consumed.
                # O2: skipped when hoisted above (num_chunks == 1).
                collective_callback = None
                if (
                    collective_callbacks_by_head is not None
                    and not hoist_collectives
                    and chunk_idx == 0
                    and q_block_idx == 0
                ):
                    collective_callback = collective_callbacks_by_head[head_idx]

                attention_const_max_core(
                    k_stream=k_streams_by_head[head_idx][chunk_idx],
                    v_stream=v_streams_by_head[head_idx][chunk_idx],
                    q_block=q_block,
                    o_aug_psums=o_aug_psums,
                    softmax_max=softmax_max,
                    softmax_scale=softmax_scale,
                    collective_callback=collective_callback,
                )

                for ot_idx in range(num_output_tiles):
                    sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                    o_tile_idx = q_block_idx * o_tiles_per_block + ot_idx
                    acc_view = o_aug_acc[:sq_tile_size, head_idx, o_tile_idx, :]
                    if chunk_idx == 0:
                        nisa.tensor_copy(dst=acc_view, src=o_aug_psums[ot_idx].data, engine=nisa.engine.vector)
                    else:
                        nisa.tensor_tensor(dst=acc_view, data1=acc_view, data2=o_aug_psums[ot_idx].data, op=nl.add)

    for head_idx in range(Nq):
        o_tiles = nt.tiles(o_hbm[head_idx], tile_size=(_O_TILE_SIZE, d))
        for o_tile_idx in range(num_o_tiles_per_head):
            sq_tile_size = min(_O_TILE_SIZE, Sq - o_tile_idx * _O_TILE_SIZE)
            o_out = _normalize(o_aug_acc[:sq_tile_size, head_idx, o_tile_idx, :])
            o_tiles[o_tile_idx, 0].store(o_out, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def attention_const_max_core(
    k_stream: "_KVBlockStream",
    v_stream: "_KVBlockStream",
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
    collective_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Compute scaled dot-product attention for one query block over all KV tiles.

    Args:
        k_stream (_KVBlockStream): Key tiles to attend over.
        v_stream (_KVBlockStream): Value tiles to attend over.
        q_block (nl.NkiTensor): [d, sq_block_size] query block.
        o_aug_psums (nt.NDSlice): Output accumulator, one [sq_tile_size, d + 1]
            tile per query slice, holding the unnormalized output and the softmax
            denominator. Written in place.
        softmax_max (Union[float, nl.NkiTensor]): Max subtracted for numerical
            stability. Scalar or [P_MAX, 1] tensor.
        softmax_scale (Optional[float]): Query-key scaling factor.
        collective_callback (Optional[Callable[[], None]]): If given, invoked once
            inside the first KV block's no_reorder region so any collective it issues
            is scheduled alongside — and overlapped with — this block's compute.

    Returns:
        None: Results are written in place into ``o_aug_psums``.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    """
    d = _P
    kernel_assert(
        (k_stream.num_blocks, k_stream.num_tiles, k_stream.last_tile_extent)
        == (v_stream.num_blocks, v_stream.num_tiles, v_stream.last_tile_extent),
        "K/V stream geometry must match",
    )
    num_blocks = k_stream.num_blocks
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]
    total_tiles = k_stream.num_tiles
    last_tile_idx = total_tiles - 1
    last_tile_extent = k_stream.last_tile_extent

    # bf16 on gen4: fp32 scores double the PSUM footprint next to the o_aug accumulators and
    # neuronx-cc fails with NCC_IGCA068 "Tried to insert PSUM spill at end of block".
    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    copy_engine = nisa.engine.scalar if _has_vector_engine_exp() else nisa.engine.vector
    # _SCORES_CLIP is in SCALED units. The vector-engine path pre-scales Q at load, so the
    # scores here are already scaled; the activation path (gen4 sub0 / A0) scales inside
    # the exp, so the scores here are raw and the clip must be raw too -- clipping raw
    # scores at 70 clipped scaled scores at 70/sqrt(d) ~ 6.2 and flattened every peak.
    scores_clip = _SCORES_CLIP if _has_vector_engine_exp() else _SCORES_CLIP / softmax_scale

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting.
    prologue_depth = min(8, total_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    num_prob_buffers = prologue_depth + 1
    probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)

    kv_tile_idx = 0  # flat index across all KV blocks
    softmax_max_is_scalar = isinstance(softmax_max, (int, float))

    for block_idx in range(num_blocks):
        k_block = k_stream.load(block_idx)[0]
        v_block = v_stream.load(block_idx)
        tiles_in_block = k_block.shape[0]

        with nl.no_reorder():
            if block_idx == 0 and collective_callback is not None:
                collective_callback()
            for local_tile in range(tiles_in_block):
                v_buf = kv_tile_idx % num_v_aug_buffers
                prob_buf = kv_tile_idx % num_prob_buffers
                tile_extent = last_tile_extent if kv_tile_idx == last_tile_idx else _P

                # v_copy(i) — capture V tile now while block buffer is valid
                nisa.tensor_copy(
                    dst=v_aug_all[0:tile_extent, v_buf, :d],
                    src=v_block[local_tile, 0].data[0:tile_extent, :],
                    engine=copy_engine,
                )

                # MM1(i): K[D, r]^T @ Q[D, Sq] -> scores[r, Sq]. The free axis of
                # stationary K becomes the PSUM partition axis, so a final r < 128
                # naturally produces a partial-partition score tile.
                scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=scores[0:tile_extent, :],
                    stationary=k_block[local_tile].data[:, 0:tile_extent],
                    moving=q_block,
                )

                # Clip to prevent inf/nan
                scores_clipped = nl.ndarray(shape=(tile_extent, sq_block_size), dtype=scores_dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=scores_clipped,
                    data=scores[0:tile_extent, :],
                    op0=nl.minimum,
                    operand0=scores_clip,
                )

                # exp(i) — scale + subtract-max + exp, over the clamped SBUF scores
                tile_softmax_max = softmax_max if softmax_max_is_scalar else softmax_max[0:tile_extent, :]
                if _has_vector_engine_exp():
                    nisa.exponential(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        src=scores_clipped,
                        max_value=tile_softmax_max,
                    )
                else:
                    nisa.activation(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        data=scores_clipped,
                        op=nl.exp,
                        scale=softmax_scale,
                        bias=tile_softmax_max,
                    )

                # MM2(i-P) — only after prologue is filled
                prologue_filled = kv_tile_idx >= prologue_depth
                if prologue_filled:
                    consume_idx = kv_tile_idx - prologue_depth
                    consume_v_buf = consume_idx % num_v_aug_buffers
                    consume_prob_buf = consume_idx % num_prob_buffers
                    should_accumulate = consume_idx != 0
                    cur_v_aug = v_aug_all[:, consume_v_buf, :]
                    consume_extent = last_tile_extent if consume_idx == last_tile_idx else _P
                    for ot_idx in range(num_output_tiles):
                        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                        nisa.nc_matmul(
                            dst=o_aug_psums[ot_idx].data,
                            stationary=probs_all[
                                0:consume_extent,
                                consume_prob_buf,
                                nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                            ],
                            moving=cur_v_aug[0:consume_extent, :],
                            accumulate=should_accumulate,
                        )

                kv_tile_idx += 1

    # === Epilogue: drain remaining prologue_depth MM2s ===
    kernel_assert(kv_tile_idx == total_tiles, f"stream yielded {kv_tile_idx} KV tiles, expected {total_tiles}")
    with nl.no_reorder():
        for i in range(prologue_depth):
            drain_idx = total_tiles - prologue_depth + i
            drain_v_buf = drain_idx % num_v_aug_buffers
            drain_prob_buf = drain_idx % num_prob_buffers
            should_accumulate = drain_idx != 0
            drain_v_aug = v_aug_all[:, drain_v_buf, :]
            drain_extent = last_tile_extent if drain_idx == last_tile_idx else _P
            for ot_idx in range(num_output_tiles):
                sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                nisa.nc_matmul(
                    dst=o_aug_psums[ot_idx].data,
                    stationary=probs_all[
                        0:drain_extent,
                        drain_prob_buf,
                        nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                    ],
                    moving=drain_v_aug[0:drain_extent, :],
                    accumulate=should_accumulate,
                )


def _normalize(o_aug_psum: nl.NkiTensor) -> nl.NkiTensor:
    """Normalize directly from PSUM: O = O_aug[:, :d] * (1 / max(O_aug[:, d], eps)).

    o_aug_psum: [sq_tile_size, d+1] in PSUM — columns 0..d-1 are weighted output, column d is sum_exp.
    Returns: [sq_tile_size, d] bf16 in SBUF.
    """
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1

    # Clamp sum_exp to avoid division by zero when all scores underflow
    sum_exp = nl.ndarray(shape=(sq_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sum_exp, data=o_aug_psum[:, d : d + 1], op0=nl.maximum, operand0=_EPS)

    inv_sum_exp = nl.ndarray(shape=(sq_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum_exp, data=sum_exp)

    o_out = nl.ndarray(shape=(sq_tile_size, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_out, data=o_aug_psum[:, :d], op0=nl.multiply, operand0=inv_sum_exp)
    return o_out


def _has_vector_engine_exp() -> bool:
    """True when nisa.exponential is available (gen4-sub1+), using the Vector engine for exp."""
    if _FORCE_ACT_EXP:
        return False
    return (nisa.get_nc_version() > nisa.nc_version.gen4) or (
        nisa.get_nc_version() == nisa.nc_version.gen4 and nisa.get_nc_sub_version() == 1
    )


class _KVBlockStream(nl.NKIObject):
    """Stream KV tile blocks with optional transpose, handling a sub-128 last tile.

    Tiles are 128×128: K tiles are [d, Sk_tile] where d=pmax and Sk_tile=pmax (Sk
    lands on the partition axis for scores), V tiles are [Sk_tile, d] for the same
    reason. Both dimensions are pmax, so tiling is always at _P.

    neurotile's BlockStream does not support a partial last tile within a block.
    This class splits it into a separate tail block so all tiles within each
    streamed block have uniform extent.

    TODO: Replace with neurotile's BlockStream once it supports partial last tiles.

    Attributes:
        num_blocks: Number of blocks to iterate over.
        num_tiles: Total number of tiles across all blocks.
        last_tile_extent: Number of active elements in the last tile (≤ _P).
    """

    def __init__(
        self,
        source: nl.NkiTensor,
        block_size: tuple,
        tiling_dim: int,
        transpose: bool = False,
    ):
        seq_len = source.shape[tiling_dim]
        tail_extent = seq_len % _P
        body_len = seq_len - tail_extent  # 128-aligned portion
        self.num_tiles = (seq_len + _P - 1) // _P
        self.last_tile_extent = tail_extent if tail_extent != 0 else _P
        self._transpose = transpose

        def blocks_over(lo, hi):
            """Create a 1D block grid over source[lo:hi] along tiling_dim."""
            idx = [slice(None)] * len(source.shape)
            idx[tiling_dim] = slice(lo, hi)
            block_grid = nt.blocks(source[tuple(idx)], tile_size=(_P, _P), block_size=block_size)
            stream_idx = [0] * len(block_size)
            stream_idx[tiling_dim] = slice(None)
            return block_grid[tuple(stream_idx)]

        tiles_per_block = block_size[tiling_dim]  # number of tiles in one block
        block_len = tiles_per_block * _P  # elements per block along tiling_dim
        self._body_block_count = (body_len + block_len - 1) // block_len

        if tail_extent == 0:
            self._body_stream = blocks_over(0, seq_len).stream(buffer_count=2, transpose=transpose)
            self._tail_block = None
        elif body_len == 0:
            self._body_stream = None
            self._body_block_count = 0
            self._tail_block = blocks_over(0, seq_len)[0]
        else:
            self._body_stream = blocks_over(0, body_len).stream(buffer_count=2, transpose=transpose)
            self._tail_block = blocks_over(body_len, seq_len)[0]

        self.num_blocks = self._body_block_count + (self._tail_block != None)

    def load(self, index: int, dge_mode=None, engine=None):
        """Load block at ``index``, returning the tile grid for this block."""
        kernel_assert(index < self.num_blocks, f"block index {index} out of range (num_blocks={self.num_blocks})")
        if index < self._body_block_count:
            return self._body_stream.load(index, transpose=self._transpose, dge_mode=dge_mode, engine=engine)
        return self._tail_block.load(transpose=self._transpose, dge_mode=dge_mode, engine=engine)


class _QBlockLoaderHeadMajor(nl.NKIObject):
    """Load Q blocks from [d, Sq] layout as scaled [d, Sq_block] SBUF tensors.

    Single bulk DMA per block via neurotile tile load.
    When ``softmax_scale`` is not None, it is fused into the load.

    Attributes:
        num_blocks: Number of Q blocks along the Sq axis.
    """

    def __init__(
        self,
        q_head_hbm: nl.NkiTensor,
        max_sq_block_size: int,  # max Sq elements per block
        softmax_scale: Optional[float] = None,
    ):
        self._softmax_scale = softmax_scale
        self._q_tile_grid = nt.tiles(q_head_hbm, tile_size=(_P, max_sq_block_size))
        self.num_blocks = self._q_tile_grid.shape[1]

    def load(self, block_idx: int) -> tuple:
        """Return ``(q_block, sq_block_size)`` with q_block as [d, sq_block_size] in SBUF."""
        loaded = self._q_tile_grid[0, block_idx].load()
        q_block = loaded.data
        sq_block_size = loaded.element_shape[1]  # actual Sq elements in this block
        if self._softmax_scale != None:
            nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=self._softmax_scale)
        return q_block, sq_block_size
