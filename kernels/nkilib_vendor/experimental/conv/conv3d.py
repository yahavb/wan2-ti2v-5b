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

"""3D Convolution kernel for NeuronCore with K-replication strategy."""

from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    div_ceil,
    get_nl_act_fn_from_type,
    get_verified_program_sharding_info,
)
from ...core.utils.logging import get_logger

_PARTITION_STRIDE_32 = 32
_PARTITION_STRIDE_64 = 64
_MAX_K_REP_AT_32 = 4
_MAX_K_REP_AT_64 = 2
_PSUM_BANK_SIZE = nl.tile_size.psum_fmax * 4
_NUM_PSUM_BANKS = 8
_BIAS_DTYPE_SIZE = sizeinbytes(nl.float32)
_HEAP_ALIGN = nl.tile_size.sbuf_min_align
_SBUF_RESERVE = 1024
# Column-tiling column sizes: the 128 PE columns slice into 4x32 (gen3+) or 2x64 (gen2+).
_COLUMN_TILE_32 = 32
_COLUMN_TILE_64 = 64


@nki.jit
def conv3d(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
    fuse_batch_norm: bool = False,
    batch_norm_eps: Optional[float] = 1e-5,
    gamma: Optional[nl.NkiTensor] = None,
    beta: Optional[nl.NkiTensor] = None,
    momentum: Optional[float] = 0.1,
    running_means: Optional[nl.NkiTensor] = None,
    running_variances: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
) -> nl.NkiTensor:
    """
    3D Convolution using tensor engine with K-replication strategy and W-contiguous tiling.

    Implements a 3D convolution operation (x_in * filters + bias) optimized for NeuronCore
    using a K-replication strategy for filter loading and W-contiguous tiling for output
    computation. Supports configurable stride, padding, dilation, optional bias, and
    optional activation function.

    Intended Usage Range:
        B: 1-128
        C_in: 3-1280, C_out: 3-2048
        D: 1-1024, H: 1-1024, W: 1-1024
        K_d: 1-64, K_h: 1-64, K_w: 1-64
        Stride: 1-64 per dimension
        Dilation: 1-64 per dimension

    Dimensions:
        B: Batch size
        C_in: Number of input channels
        C_out: Number of output channels
        D: Input depth
        H: Input height
        W: Input width
        K_d: Filter depth
        K_h: Filter height
        K_w: Filter width
        D_out: Output depth = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
        H_out: Output height = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
        W_out: Output width = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1

    Args:
        x_in (nl.NkiTensor): [B, C_in, D, H, W], Input tensor on HBM.
        filters (nl.NkiTensor): [K_d, K_h, K_w, C_in, C_out], Filter weights on HBM.
        bias (Optional[nl.NkiTensor]): [C_out], Optional bias tensor on HBM.
        stride (tuple[int, int, int]): (stride_d, stride_h, stride_w), Convolution strides.
        padding (tuple[int, int, int, int, int, int]): (pad_d_left, pad_d_right, pad_h_top,
            pad_h_bottom, pad_w_left, pad_w_right), Padding for each spatial dimension.
        dilation (tuple[int, int, int]): (dilation_d, dilation_h, dilation_w), Dilation factors.
        activation_fn (Optional[ActFnType]): Optional activation function to apply after conv.
            When fuse_batch_norm is True, the activation is instead applied after the fused
            batchnorm (statistics are still computed on the raw convolution output).
        lnc_shard (bool): Does nothing. Will be deprecated in next release.
        sbm (Optional[SbufManager]): Optional caller-provided SBUF manager for allocation. When
            provided, its allocation mode must match use_auto_allocation.
        use_auto_allocation (bool): Whether to use auto allocation for SBUF buffers.
        fuse_batch_norm (bool): When True, additionally compute the per-C_out-channel
            mean and variance of the convolution output (reduced over B, D_out, H_out,
            W_out) and apply a fused batchnorm2D (equivalent to torch.nn.BatchNorm2d)
            to the output using those statistics. May be combined with activation_fn, in
            which case the activation is applied to the normalized output.
            May also be combined with lnc_shard: the per-channel statistics are reduced
            across the two neuron cores (a cross-core reduction of the partial sums via
            nisa.sendrecv when the work is sharded on the D-H spatial dims, or each core
            owning a disjoint C_out slice end-to-end otherwise, so no cross-core read is
            needed).
        batch_norm_eps (Optional[float]): Epsilon added to the variance for numerical
            stability when fuse_batch_norm is True. Default 1e-5.
        gamma (Optional[nl.NkiTensor]): [C_out, 1], per-channel batchnorm scale. Required
            when fuse_batch_norm is True.
        beta (Optional[nl.NkiTensor]): [C_out, 1], per-channel batchnorm shift. Required
            when fuse_batch_norm is True.
        momentum (Optional[float]): Momentum for the running-statistics update when
            fuse_batch_norm is True, matching torch.nn.BatchNorm2d:
            x_running = (1 - momentum) * x_running + momentum * x_new. Default 0.1.
        running_means (Optional[nl.NkiTensor]): [C_out, 1], float32 running mean carried
            across steps. Required when fuse_batch_norm is True.
        running_variances (Optional[nl.NkiTensor]): [C_out, 1], float32 running variance
            carried across steps. Required when fuse_batch_norm is True.

    Returns:
        y_out (nl.NkiTensor): [B, C_out, D_out, H_out, W_out], Output tensor on HBM.
        When fuse_batch_norm is True, returns a tuple (y_out, means, variances,
        updated_running_means, updated_running_variances) where y_out has the batchnorm
        applied. means and variances are [C_out] float32 tensors on HBM holding the raw
        (pre-normalization) convolution-output statistics (variance with correction 0).
        updated_running_means and updated_running_variances are [C_out, 1] float32 tensors
        on shared HBM holding the momentum-updated running statistics; the variance update
        uses the correction-1 (unbiased) variance, matching torch.nn.BatchNorm2d.

    Pseudocode:
        cfg = build_config(x_in, filters, bias, stride, padding, dilation)
        tile_cfg = build_tile_config(cfg)
        mem_cfg = build_memory_config(cfg, tile_cfg)
        allocate SBUF buffers (filters, bias, results, input windows, stacked inputs)

        for c_out_group in range(0, C_out, c_out_interleave * P_MAX):
            load filters + bias for c_out_group into SBUF

            for batch in range(B):
                for dh_group in range(dh_start, dh_end, num_dh_stacked):
                    for w_tile in range(0, W_out, W_tile):
                        allocate PSUM banks for output tiles

                        for c_in_tile in range(0, C_in, P_MAX):
                            DMA load input window [C_in_tile, D_win, H_win, W_win] from HBM to SBUF
                            for k_outer_tile in range(K_outer_tile_count):
                                scatter input window to stacked layout via tensor_copy
                                nc_matmul: stationary=filters, moving=stacked_input, dst=PSUM

                        apply bias + activation, copy PSUM to result SBUF
                        DMA store result SBUF to HBM

        free SBUF buffers

    Tiling Strategy:
        Loop nest: C_out groups -> Batch -> D-H groups -> W tiles -> (C_in tiles in _conv3d_output_tile)

        SBUF Memory Layout:
        - Filters: [c_out_interleave] x [c_in_tile_count] x [K_outer_tile_count] buffers,
          each (stacked_filter_dim_max, c_out_tile_size_max). Loaded once per C_out group.
        - Bias: [c_out_interleave] buffers, (c_out_tile_size_max, 1), float32.
          Loaded once per C_out group alongside filters.
        - Result SBUF: [w_out_interleave] x [c_out_interleave] buffers,
          each (c_out_tile_size_max, num_dh_stacked * W_tile).
          Rotated across W tiles for pipelining DMA stores with compute.
        - Input windows: [input_window_interleave] buffers,
          each (c_in_tile_size_max, d_window_max, h_window_max, w_window_max).
          Multi-buffered to overlap HBM loads with SBUF scatter.
        - Stacked inputs: [stacked_input_interleave] x [K_outer_tile_count] buffers,
          each (stacked_filter_dim_max, num_dh_stacked * W_tile).
          Multi-buffered to overlap scatter with matmul.

        Engine pipelining: tensor_copy and memset alternate between vector/scalar and
        gpsimd/vector engines respectively to enable concurrent execution.

        Tile Size Selection:
        - W_tile: min(W_out, F_MAX), reduced if memory does not fit
        - num_dh_stacked: packs multiple (d_out, h_out) on free dim when W_out < F_MAX
        - c_out/c_in tile: min(P_MAX, channel_count)
        - K_REP: replicates filter positions within partition stride for small C_in tiles
        - All sizes iteratively reduced in _build_tile_config until SBUF budget is met
    """
    if lnc_shard:
        get_logger("conv3d").warn(
            "lnc_shard does nothing and will be deprecated in the next release. "
            "Please stop passing lnc_shard to conv3d."
        )

    cfg = _build_conv3d_config(
        x_in, filters, bias, stride, padding, dilation, activation_fn, lnc_shard, fuse_batch_norm
    )
    dtype_size = sizeinbytes(x_in.dtype)

    if sbm != None:
        kernel_assert(
            sbm.is_auto_alloc() == use_auto_allocation,
            f"[conv3d] If an SbufManager is provided, sbm.is_auto_alloc() ({sbm.is_auto_alloc()}) "
            f"must equal use_auto_allocation ({use_auto_allocation}).",
        )
        if use_auto_allocation:
            total_sbuf_budget = nl.tile_size.total_available_sbuf_size
        else:
            total_sbuf_budget = sbm.get_free_space()
    else:
        total_sbuf_budget = nl.tile_size.total_available_sbuf_size

    tile_cfg = _build_tile_config(cfg, dtype_size, total_sbuf_budget)
    mem_cfg = _build_memory_config(cfg, tile_cfg, dtype_size, total_sbuf_budget)
    _validate_conv3d_inputs(x_in, filters, bias, cfg, gamma, beta, running_means, running_variances)

    y_out = nl.ndarray(
        shape=(cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
        dtype=x_in.dtype,
        buffer=nl.shared_hbm,
    )

    # Separate conv_out (raw) / y_out (normalized) buffers: in-place HBM RMW is unsafe under LNC=2
    # redundant execution. See spec.
    means = None
    variances = None
    updated_running_means = None
    updated_running_variances = None
    bn_sum_bufs = None
    bn_sqsum_bufs = None
    conv_out = y_out
    # SPMD sharding info: n_prgs = core count, prg_id = this core (0/1 at LNC=2).
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))
    if cfg.fuse_batch_norm:
        means = nl.ndarray(shape=(cfg.C_out,), dtype=nl.float32, buffer=nl.shared_hbm)
        variances = nl.ndarray(shape=(cfg.C_out,), dtype=nl.float32, buffer=nl.shared_hbm)
        updated_running_means = nl.ndarray(shape=(cfg.C_out, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        updated_running_variances = nl.ndarray(shape=(cfg.C_out, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        conv_out = nl.ndarray(
            shape=(cfg.B, cfg.C_out, cfg.D_out, cfg.H_out, cfg.W_out),
            dtype=x_in.dtype,
            buffer=nl.shared_hbm,
        )

    caller_provided_sbm = sbm != None
    if not caller_provided_sbm:
        sbm = SbufManager(0, mem_cfg.total_memory_required, logger=get_logger("conv3d"))

    # Batchnorm accumulators live for the whole kernel (allocated first, freed last).
    # Layout: partition = channel within a C_out tile, column = global C_out tile index.
    bn_sq_scratch = None
    bn_sqsum_partials = None
    if cfg.fuse_batch_norm:
        full_c_out_tile_count = div_ceil(cfg.C_out, tile_cfg.P_MAX)
        # Column tiling: band g writes partials at [g * col_size, ...), so accumulators are
        # C_OUT_REP * col_size partitions tall.
        bn_acc_part_dim = (
            tile_cfg.C_OUT_REP * tile_cfg.col_size if tile_cfg.C_OUT_REP > 1 else tile_cfg.c_out_tile_size_max
        )
        bn_sum_bufs = sbm.alloc_heap(
            shape=(bn_acc_part_dim, full_c_out_tile_count),
            dtype=nl.float32,
            name="bn_sum_acc",
        )
        bn_sqsum_bufs = sbm.alloc_heap(
            shape=(bn_acc_part_dim, full_c_out_tile_count),
            dtype=nl.float32,
            name="bn_sqsum_acc",
        )
        # Throwaway destination for the squared activation; only its fused reduction is kept.
        bn_sq_scratch = sbm.alloc_heap(
            shape=(bn_acc_part_dim, tile_cfg.dh_head_size * tile_cfg.W_tile),
            dtype=nl.float32,
            name="bn_sq_scratch",
        )
        # Per-flush sum(Y^2) partials: column = global_tile_idx * bn_partial_iters + iter_idx.
        # Reduced in the finalize pass (no RMW here).
        bn_sqsum_partials = sbm.alloc_heap(
            shape=(bn_acc_part_dim, full_c_out_tile_count * tile_cfg.bn_partial_iters),
            dtype=nl.float32,
            name="bn_sqsum_partials",
        )
        # Pre-zero sum(Y) (accumulated in place). Under column tiling a flush may leave partials
        # columns unwritten, so pre-zero those too.
        nisa.memset(dst=bn_sum_bufs[:, :], value=0.0)
        if tile_cfg.C_OUT_REP > 1:
            nisa.memset(dst=bn_sqsum_partials[:, :], value=0.0)

    pre_bias_bufs, pre_filter_bufs = _allocate_filter_bias_buffers(sbm, cfg, tile_cfg, mem_cfg, x_in.dtype)

    max_effective_free = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    # Column tiling: result buffer is C_OUT_REP bands (band g at g * col_size, dh_head_size*W_tile
    # wide); rep == 1 keeps the (c_out_tile, full-free) shape.
    if tile_cfg.C_OUT_REP > 1:
        result_part_dim = tile_cfg.C_OUT_REP * tile_cfg.col_size
        band_free = tile_cfg.dh_head_size * tile_cfg.W_tile
    else:
        result_part_dim = tile_cfg.c_out_tile_size_max
        band_free = max_effective_free
    # Store-batch merging: a slot holds store_batch_merge batches side by side (batch b at cols
    # [b * band_free, +band_free)); store_batch_merge * store_pipe == w_out_interleave.
    result_free_dim = mem_cfg.store_batch_merge * band_free
    result_sbuf_slots = []
    for w_slot_idx in range(mem_cfg.store_pipe):
        slot = []
        for c_out_slot_idx in range(mem_cfg.c_out_interleave):
            slot.append(
                sbm.alloc_heap(
                    shape=(result_part_dim, result_free_dim),
                    dtype=x_in.dtype,
                    name=f"result_{w_slot_idx}_cout{c_out_slot_idx}",
                )
            )
        result_sbuf_slots.append(slot)

    input_window_slots = []
    for input_window_slot_idx in range(mem_cfg.input_window_interleave):
        input_window_slots.append(
            sbm.alloc_heap(
                shape=(
                    tile_cfg.c_in_tile_size_max,
                    tile_cfg.d_window_max,
                    tile_cfg.h_window_max,
                    tile_cfg.w_window_max,
                ),
                dtype=x_in.dtype,
                name=f"input_window_{input_window_slot_idx}",
            )
        )

    stacked_input_slots = []
    for stacked_input_slot_idx in range(mem_cfg.stacked_input_interleave):
        stacked_input_bufs = []
        for k_outer_idx in range(tile_cfg.K_outer_tile_count):
            stacked_input_bufs.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, max_effective_free),
                    dtype=x_in.dtype,
                    name=f"stacked_input_{stacked_input_slot_idx}_k{k_outer_idx}",
                )
            )
        stacked_input_slots.append(stacked_input_bufs)

    for c_out_group_start in range(tile_cfg.C_out_start, tile_cfg.C_out_end, mem_cfg.c_out_interleave * tile_cfg.P_MAX):
        c_out_group_end = min(c_out_group_start + mem_cfg.c_out_interleave * tile_cfg.P_MAX, tile_cfg.C_out_end)
        actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)

        c_out_tile_sizes, bias_cache, filters_cache = _load_bias_and_filters_for_c_out_group(
            filters,
            bias,
            cfg,
            tile_cfg,
            c_out_group_start,
            c_out_group_end,
            pre_bias_bufs,
            pre_filter_bufs,
        )

        w_out_tile_counter = 0
        c_in_slot_counter = 0

        if mem_cfg.store_batch_merge > 1:
            # Store-batch-merge path: one batch-strided DMA per band over store_batch_merge
            # batches. Column-tiling only (which disables H-chunking).
            w_out_tile_counter, c_in_slot_counter = _conv3d_c_out_group_merged_store(
                x_in=x_in,
                conv_out=conv_out,
                filters_cache=filters_cache,
                bias_cache=bias_cache,
                result_sbuf_slots=result_sbuf_slots,
                input_window_slots=input_window_slots,
                stacked_input_slots=stacked_input_slots,
                cfg=cfg,
                tile_cfg=tile_cfg,
                mem_cfg=mem_cfg,
                c_out_tile_sizes=c_out_tile_sizes,
                actual_group_size=actual_group_size,
                c_out_group_start=c_out_group_start,
                bn_sum_bufs=bn_sum_bufs,
                bn_sq_scratch=bn_sq_scratch,
                bn_sqsum_partials=bn_sqsum_partials,
            )
            continue

        for batch_idx in range(cfg.B):
            x_in_batch = x_in.select(dim=0, index=batch_idx)

            chunk_preloaded_windows = None
            dh_group_idx = tile_cfg.dh_start
            while dh_group_idx < tile_cfg.dh_end:
                dh_positions = []
                first_d_out = None
                for dh_stacked_idx in range(tile_cfg.num_dh_stacked):
                    flat_idx = dh_group_idx + dh_stacked_idx
                    if flat_idx >= tile_cfg.dh_end:
                        break
                    d_out_idx, h_out_idx = divmod(flat_idx, cfg.H_out)
                    if first_d_out == None:
                        first_d_out = d_out_idx
                    elif d_out_idx != first_d_out:
                        break  # don't cross D boundary — avoids window spanning full H
                    dh_positions.append((d_out_idx, h_out_idx))
                num_dh_positions = len(dh_positions)

                # H-chunking participation for this dh position.
                d_out_current = dh_positions[0][0]
                h_out_current = dh_positions[0][1]
                h_chunk_enabled = tile_cfg.h_chunk_size > 1 and tile_cfg.num_dh_stacked == 1
                is_chunk_start = h_chunk_enabled and (h_out_current % tile_cfg.h_chunk_size == 0)

                # Build the chunk's dh_positions
                if is_chunk_start:
                    chunk_dh_positions = []
                    for h_chunk_idx in range(tile_cfg.h_chunk_size):
                        chunk_h_out = h_out_current + h_chunk_idx
                        if chunk_h_out >= cfg.H_out:
                            break
                        # Don't cross the d_out boundary.
                        chunk_flat = d_out_current * cfg.H_out + chunk_h_out
                        if chunk_flat >= tile_cfg.dh_end:
                            break
                        chunk_dh_positions.append((d_out_current, chunk_h_out))

                for w_start in range(0, cfg.W_out, tile_cfg.W_tile):
                    w_end = min(w_start + tile_cfg.W_tile, cfg.W_out)
                    w_tile_size = w_end - w_start

                    # At an H-chunk start, preload the chunk's union input window for all c_in tiles.
                    if is_chunk_start:
                        chunk_preloaded_windows = []
                        c_in_tile_idx_load = 0
                        for c_in_start_load in range(0, cfg.C_in, tile_cfg.P_MAX):
                            c_in_end_load = min(c_in_start_load + tile_cfg.P_MAX, cfg.C_in)
                            global_idx_load = c_in_slot_counter + c_in_tile_idx_load
                            input_window_buf_load = input_window_slots[global_idx_load % len(input_window_slots)]
                            x_in_cin_load = x_in_batch.slice(dim=0, start=c_in_start_load, end=c_in_end_load, step=1)
                            loaded = _load_input_window_to_sbuf_3d(
                                x_in_cin=x_in_cin_load,
                                input_window_buf=input_window_buf_load,
                                cfg=cfg,
                                dh_positions=chunk_dh_positions,
                                w_start=w_start,
                                w_end=w_end,
                            )
                            chunk_preloaded_windows.append(loaded)
                            c_in_tile_idx_load += 1

                    slot_idx = w_out_tile_counter % mem_cfg.w_out_interleave
                    result_sbufs = result_sbuf_slots[slot_idx][:actual_group_size]
                    psum_bank_idx = (slot_idx * actual_group_size) % _NUM_PSUM_BANKS

                    # Preloaded windows only when H-chunking.
                    preloaded = chunk_preloaded_windows if h_chunk_enabled else None

                    _conv3d_output_tile(
                        x_in_batch=x_in_batch,
                        filters_cache=filters_cache,
                        bias_cache=bias_cache,
                        result_sbufs=result_sbufs,
                        input_window_slots=input_window_slots,
                        stacked_input_slots=stacked_input_slots,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                        psum_bank_idx=psum_bank_idx,
                        c_in_slot_offset=c_in_slot_counter,
                        preloaded_windows=preloaded,
                        bn_c_out_tile_base=c_out_group_start // tile_cfg.P_MAX,
                        bn_sum_acc=bn_sum_bufs,
                        bn_sq_scratch=bn_sq_scratch,
                        bn_sqsum_partials=bn_sqsum_partials,
                        bn_partial_iters=tile_cfg.bn_partial_iters,
                        bn_iter_idx=w_out_tile_counter,
                    )
                    c_in_slot_counter += tile_cfg.c_in_tile_count

                    # Store SBUF -> HBM, one DMA per column-tiling band (band g: dh [g * band_dh,
                    # +band_dh) at partitions [g * col_size, +M)); col_rep == 1 is a single band.
                    total_dh = cfg.D_out * cfg.H_out
                    col_rep_st, band_dh_st, col_size_st = _flush_column_layout(tile_cfg, num_dh_positions)
                    store_bands = _column_bands(num_dh_positions, band_dh_st, col_rep_st, col_size_st)
                    for c_out_tile_idx in range(actual_group_size):
                        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
                        c_out_tile_end = c_out_tile_start + c_out_tile_sizes[c_out_tile_idx]
                        for band_idx in range(len(store_bands)):
                            part_off = store_bands[band_idx][0]
                            band_dh_lo = store_bands[band_idx][1]
                            band_dh_hi = store_bands[band_idx][2]
                            this_band_dh = band_dh_hi - band_dh_lo
                            result_view = (
                                (result_sbufs[c_out_tile_idx])
                                .slice(dim=0, start=part_off, end=part_off + c_out_tile_sizes[c_out_tile_idx], step=1)
                                .slice(dim=1, start=0, end=this_band_dh * w_tile_size, step=1)
                                .reshape_dim(1, (this_band_dh, w_tile_size))
                            )
                            y_out_view = (
                                conv_out.select(dim=0, index=batch_idx)
                                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                                .flatten_dims(1, 3)
                                .reshape_dim(1, (total_dh, cfg.W_out))
                                .slice(
                                    dim=1,
                                    start=dh_group_idx + band_dh_lo,
                                    end=dh_group_idx + band_dh_hi,
                                    step=1,
                                )
                                .slice(dim=2, start=w_start, end=w_end, step=1)
                            )
                            nisa.dma_copy(
                                dst=y_out_view,
                                src=result_view,
                                dge_mode=nisa.dge_mode.hwdge,
                                engine=nki.isa.engine.sync,
                            )

                    w_out_tile_counter += 1
                dh_group_idx += num_dh_positions

    # Free all heap allocations
    for _ in range(mem_cfg.stacked_input_interleave):
        for _ in range(tile_cfg.K_outer_tile_count):
            sbm.pop_heap()
    for _ in range(mem_cfg.input_window_interleave):
        sbm.pop_heap()
    for _ in range(mem_cfg.store_pipe):
        for _ in range(mem_cfg.c_out_interleave):
            sbm.pop_heap()
    for _ in range(tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count):
        sbm.pop_heap()
    for _ in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            sbm.pop_heap()

    if cfg.fuse_batch_norm:
        _finalize_batch_norm(
            cfg,
            tile_cfg,
            sbm,
            conv_out,
            y_out,
            gamma,
            beta,
            batch_norm_eps,
            bn_sum_bufs,
            bn_sqsum_bufs,
            bn_sqsum_partials,
            means,
            variances,
            n_prgs,
            prg_id,
            momentum,
            running_means,
            running_variances,
            updated_running_means,
            updated_running_variances,
            mem_cfg.bn_reload_interleave,
        )
        # Free batchnorm accumulators (allocated first, so popped last).
        sbm.pop_heap()  # bn_sqsum_partials
        sbm.pop_heap()  # bn_sq_scratch
        sbm.pop_heap()  # bn_sqsum_acc
        sbm.pop_heap()  # bn_sum_acc

    if cfg.fuse_batch_norm:
        return y_out, means, variances, updated_running_means, updated_running_variances
    return y_out


def _conv3d_c_out_group_merged_store(
    x_in: nl.NkiTensor,
    conv_out: nl.NkiTensor,
    filters_cache: list,
    bias_cache: list,
    result_sbuf_slots: list,
    input_window_slots: list,
    stacked_input_slots: list,
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    mem_cfg: "Conv3dMemoryConfig",
    c_out_tile_sizes: list[int],
    actual_group_size: int,
    c_out_group_start: int,
    bn_sum_bufs: Optional[nl.NkiTensor],
    bn_sq_scratch: Optional[nl.NkiTensor],
    bn_sqsum_partials: Optional[nl.NkiTensor],
) -> tuple[int, int]:
    """
    Process one C_out group with store-batch merging (column tiling only).

    Reorders the loop nest so batch is inner to each (D-H group, W tile) flush: for a group of
    store_batch_merge consecutive batches, each batch's conv output is computed into its own
    free-column slice [b * band_free, (b + 1) * band_free) of one wide result slot, then the
    whole group is written to HBM with a single batch-strided DMA per column-tiling band —
    cutting the store-trigger count by store_batch_merge. Compute per batch (via
    _conv3d_output_tile) and the batchnorm accumulation are unchanged from the per-batch path;
    only the store is merged.

    Enabled (by _build_memory_config) only when every flush is full: D_out == 1 and the per-core
    D-H range divides evenly into num_dh_stacked, so num_dh_positions == num_dh_stacked for every
    flush and each batch's result is exactly band_free wide. Returns the updated
    (w_out_tile_counter, c_in_slot_counter) — but they are group-local here, so the caller only
    uses them for parity with the per-batch path.

    The store DMA lowers because the batch axis is an inner free stride and the channel axis (M)
    is the partition-aligned outer dim (verified against the device tracer); this is unlike the
    cross-band merge, which cannot align band-major SBUF partitions to channel-major HBM.
    """
    merge = mem_cfg.store_batch_merge
    band_free = tile_cfg.dh_head_size * tile_cfg.W_tile
    total_dh = cfg.D_out * cfg.H_out
    P_MAX = tile_cfg.P_MAX
    w_out_tile_counter = 0
    c_in_slot_counter = 0
    store_group_idx = 0

    dh_group_idx = tile_cfg.dh_start
    while dh_group_idx < tile_cfg.dh_end:
        # Enable gate guarantees full flushes: the group is num_dh_stacked consecutive positions.
        dh_positions = []
        for dh_stacked_idx in range(tile_cfg.num_dh_stacked):
            flat_idx = dh_group_idx + dh_stacked_idx
            d_out_idx, h_out_idx = divmod(flat_idx, cfg.H_out)
            dh_positions.append((d_out_idx, h_out_idx))
        num_dh_positions = len(dh_positions)
        col_rep_st, band_dh_st, col_size_st = _flush_column_layout(tile_cfg, num_dh_positions)
        store_bands = _column_bands(num_dh_positions, band_dh_st, col_rep_st, col_size_st)

        for w_start in range(0, cfg.W_out, tile_cfg.W_tile):
            w_end = min(w_start + tile_cfg.W_tile, cfg.W_out)
            w_tile_size = w_end - w_start

            # Per batch-group: stage its batches into one rotating result slot, then merged-store.
            for batch_group_start in range(0, cfg.B, merge):
                batch_group_end = min(batch_group_start + merge, cfg.B)
                group_batches = batch_group_end - batch_group_start

                slot_idx = store_group_idx % mem_cfg.store_pipe
                result_slot = result_sbuf_slots[slot_idx][:actual_group_size]

                # Compute each batch into its own free-column slice [b_local * band_free, ...).
                for b_local in range(group_batches):
                    batch_idx = batch_group_start + b_local
                    x_in_batch = x_in.select(dim=0, index=batch_idx)
                    # Per-batch result views into this slot. Explicit loop — tracer rejects comprehensions.
                    result_sbufs = []
                    for c_out_tile_idx in range(actual_group_size):
                        result_sbufs.append(
                            result_slot[c_out_tile_idx].slice(
                                dim=1, start=b_local * band_free, end=b_local * band_free + band_free, step=1
                            )
                        )
                    psum_bank_idx = (w_out_tile_counter * actual_group_size) % _NUM_PSUM_BANKS
                    _conv3d_output_tile(
                        x_in_batch=x_in_batch,
                        filters_cache=filters_cache,
                        bias_cache=bias_cache,
                        result_sbufs=result_sbufs,
                        input_window_slots=input_window_slots,
                        stacked_input_slots=stacked_input_slots,
                        cfg=cfg,
                        tile_cfg=tile_cfg,
                        c_out_tile_sizes=c_out_tile_sizes,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                        psum_bank_idx=psum_bank_idx,
                        c_in_slot_offset=c_in_slot_counter,
                        preloaded_windows=None,
                        bn_c_out_tile_base=c_out_group_start // P_MAX,
                        bn_sum_acc=bn_sum_bufs,
                        bn_sq_scratch=bn_sq_scratch,
                        bn_sqsum_partials=bn_sqsum_partials,
                        bn_partial_iters=tile_cfg.bn_partial_iters,
                        bn_iter_idx=w_out_tile_counter,
                    )
                    c_in_slot_counter += tile_cfg.c_in_tile_count
                    w_out_tile_counter += 1

                # Merged store: one batch-strided DMA per (c_out tile, band). SBUF src strides batch
                # outer (band_free apart); HBM dst permutes channel (M) outermost to align partitions.
                for c_out_tile_idx in range(actual_group_size):
                    c_out_tile_start = c_out_group_start + c_out_tile_idx * P_MAX
                    c_out_tile_end = c_out_tile_start + c_out_tile_sizes[c_out_tile_idx]
                    M = c_out_tile_sizes[c_out_tile_idx]
                    for band_idx in range(len(store_bands)):
                        part_off = store_bands[band_idx][0]
                        band_dh_lo = store_bands[band_idx][1]
                        band_dh_hi = store_bands[band_idx][2]
                        this_band_dh = band_dh_hi - band_dh_lo
                        # SBUF src -> (M, group_batches, this_band_dh, w_tile_size).
                        src_view = (
                            result_slot[c_out_tile_idx]
                            .slice(dim=0, start=part_off, end=part_off + M, step=1)
                            .slice(dim=1, start=0, end=group_batches * band_free, step=1)
                            .reshape_dim(1, (group_batches, band_free))
                            .slice(dim=2, start=0, end=this_band_dh * w_tile_size, step=1)
                            .reshape_dim(2, (this_band_dh, w_tile_size))
                        )
                        # HBM dst: conv_out[batch_group, c_out_tile, dh_band, w], channel permuted
                        # outermost to align with the SBUF partition dim.
                        dst_view = (
                            conv_out.slice(dim=0, start=batch_group_start, end=batch_group_end, step=1)
                            .slice(dim=1, start=c_out_tile_start, end=c_out_tile_end, step=1)
                            .flatten_dims(2, 4)
                            .reshape_dim(2, (total_dh, cfg.W_out))
                            .slice(dim=2, start=dh_group_idx + band_dh_lo, end=dh_group_idx + band_dh_hi, step=1)
                            .slice(dim=3, start=w_start, end=w_end, step=1)
                            .permute((1, 0, 2, 3))
                        )
                        nisa.dma_copy(
                            dst=dst_view,
                            src=src_view,
                            dge_mode=nisa.dge_mode.hwdge,
                            engine=nki.isa.engine.sync,
                        )
                store_group_idx += 1
        dh_group_idx += num_dh_positions

    return w_out_tile_counter, c_in_slot_counter


def _momentum_update_stat(
    running_hbm: nl.NkiTensor,
    new_stat_col: nl.NkiTensor,
    updated_hbm: nl.NkiTensor,
    running_scratch: nl.NkiTensor,
    new_scratch: nl.NkiTensor,
    c_out_tile_start: int,
    c_out_tile_end: int,
    one_minus_momentum: float,
    momentum: float,
    new_scale: float,
) -> None:
    """
    Momentum-update one running statistic for a C_out tile (matches torch.nn.BatchNorm2d):
        updated = one_minus_momentum * running + momentum * (new_scale * new_stat)
    new_scale folds in any per-stat factor (1.0 for the mean, the correction-1 factor N/(N-1) for
    the variance) so both stats share one path. running_scratch / new_scratch are two disjoint
    SBUF scratch columns.
    """
    nisa.dma_copy(dst=running_scratch, src=running_hbm.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
    nisa.tensor_scalar(dst=running_scratch, data=running_scratch, op0=nl.multiply, operand0=one_minus_momentum)
    nisa.tensor_scalar(
        dst=new_scratch, data=new_stat_col, op0=nl.multiply, operand0=new_scale, op1=nl.multiply, operand1=momentum
    )
    nisa.tensor_tensor(dst=running_scratch, data1=running_scratch, data2=new_scratch, op=nl.add)
    nisa.dma_copy(dst=updated_hbm.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1), src=running_scratch)


def _finalize_stats_phase(
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    sbm: SbufManager,
    bn_sum_bufs: nl.NkiTensor,
    bn_sqsum_bufs: nl.NkiTensor,
    bn_sqsum_partials: nl.NkiTensor,
    recv_sum: nl.NkiTensor,
    recv_sqsum: nl.NkiTensor,
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    identity_mask: list,
    shard_stats_on_dh: bool,
    prg_id: int,
    inv_n: float,
) -> None:
    """
    Finalize-stats phase: produce the complete per-channel mean / variance in HBM and leave them
    resident in means_sbuf / variances_sbuf for the later phases. Uses var = E[Y^2] - E[Y]^2
    (correction 0) from the accumulated sum(Y) / sum(Y^2). See conv3d_design_spec.md.

    Under shard_on_dh each core holds only *partial* sums (its D-H subset), so the cores exchange
    partials with nisa.sendrecv and add before computing stats; otherwise each core already owns
    the complete sum for its own C_out slice.
    """
    P_MAX = tile_cfg.P_MAX
    C_OUT_REP = tile_cfg.C_OUT_REP

    # Reduce per-flush sum(Y^2) partials into bn_sqsum_bufs: collapse each C_out tile's partial
    # columns into one. Only tiles this core produced are initialized (C_out-shard: own slice).
    bn_partial_iters = tile_cfg.bn_partial_iters
    reduce_tile_start = tile_cfg.C_out_start // P_MAX
    reduce_tile_end = div_ceil(tile_cfg.C_out_end, P_MAX)
    # Under column tiling reduce the full band-tall range (C_OUT_REP bands, folded below);
    # rep == 1 reduces only [0, M).
    reduce_part_dim = tile_cfg.C_OUT_REP * tile_cfg.col_size if tile_cfg.C_OUT_REP > 1 else None
    for c_out_tile_idx in range(reduce_tile_start, reduce_tile_end):
        c_out_tile_start = c_out_tile_idx * P_MAX
        c_out_tile_size = min(P_MAX, cfg.C_out - c_out_tile_start)
        part_hi = reduce_part_dim if reduce_part_dim != None else c_out_tile_size
        block_start = c_out_tile_idx * bn_partial_iters
        block_end = block_start + bn_partial_iters
        nisa.tensor_reduce(
            bn_sqsum_bufs[:part_hi, c_out_tile_idx : c_out_tile_idx + 1],
            nl.add,
            bn_sqsum_partials[:part_hi, block_start:block_end],
            axis=1,
        )

    # Column-tiling cross-band collapse: gather each band's column to [0, M) via nc_stream_shuffle
    # (band offsets g * col_size are quadrant boundaries) and sum, restoring canonical [0, M) stats.
    if tile_cfg.C_OUT_REP > 1:
        col_size = tile_cfg.col_size
        gather_sum = sbm.alloc_heap(shape=(P_MAX, C_OUT_REP), dtype=nl.float32, name="bn_gather_sum")
        gather_sqsum = sbm.alloc_heap(shape=(P_MAX, C_OUT_REP), dtype=nl.float32, name="bn_gather_sqsum")
        for c_out_tile_idx in range(reduce_tile_start, reduce_tile_end):
            c_out_tile_start = c_out_tile_idx * P_MAX
            c_out_tile_size = min(P_MAX, cfg.C_out - c_out_tile_start)
            for band_idx in range(C_OUT_REP):
                part_lo = band_idx * col_size
                nisa.nc_stream_shuffle(
                    dst=gather_sum[:c_out_tile_size, band_idx : band_idx + 1],
                    src=bn_sum_bufs[part_lo : part_lo + c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1],
                    shuffle_mask=identity_mask,
                )
                nisa.nc_stream_shuffle(
                    dst=gather_sqsum[:c_out_tile_size, band_idx : band_idx + 1],
                    src=bn_sqsum_bufs[part_lo : part_lo + c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1],
                    shuffle_mask=identity_mask,
                )
            nisa.tensor_reduce(
                bn_sum_bufs[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1],
                nl.add,
                gather_sum[:c_out_tile_size, 0:C_OUT_REP],
                axis=1,
            )
            nisa.tensor_reduce(
                bn_sqsum_bufs[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1],
                nl.add,
                gather_sqsum[:c_out_tile_size, 0:C_OUT_REP],
                axis=1,
            )
        sbm.pop_heap()  # bn_gather_sqsum
        sbm.pop_heap()  # bn_gather_sum

    if shard_stats_on_dh:
        # Each core holds a partial sum. sendrecv-exchange with the peer and add; a core_barrier +
        # peer HBM read would race (barrier only fences gpsimd). Full partials sit in [0, cm).
        cm = tile_cfg.c_out_tile_size_max
        peer = 1 - prg_id
        nisa.sendrecv(
            src=bn_sum_bufs[:cm, :],
            dst=recv_sum[:cm, :],
            send_to_rank=peer,
            recv_from_rank=peer,
            pipe_id=0,
        )
        nisa.sendrecv(
            src=bn_sqsum_bufs[:cm, :],
            dst=recv_sqsum[:cm, :],
            send_to_rank=peer,
            recv_from_rank=peer,
            pipe_id=1,
        )
        nisa.tensor_tensor(dst=bn_sum_bufs[:cm, :], data1=bn_sum_bufs[:cm, :], data2=recv_sum[:cm, :], op=nl.add)
        nisa.tensor_tensor(dst=bn_sqsum_bufs[:cm, :], data1=bn_sqsum_bufs[:cm, :], data2=recv_sqsum[:cm, :], op=nl.add)

        # Every core now has complete sums; compute stats for all C_out tiles.
        stat_c_out_start, stat_c_out_end = 0, cfg.C_out
    else:
        # Each core already owns the complete sum for its own C_out slice.
        stat_c_out_start, stat_c_out_end = tile_cfg.C_out_start, tile_cfg.C_out_end

    for c_out_tile_start in range(stat_c_out_start, stat_c_out_end, P_MAX):
        c_out_tile_idx = c_out_tile_start // P_MAX
        c_out_tile_end = min(c_out_tile_start + P_MAX, stat_c_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start

        sum_col = bn_sum_bufs[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        sqsum_col = bn_sqsum_bufs[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        # Final stats -> persistent SBUF columns (read by later phases, no reload) + HBM outputs.
        mean_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        var_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]

        # mean = sum / N, var = sum(Y^2) / N - (sum / N)^2  (E[Y^2] - E[Y]^2, correction 0)
        nisa.activation(dst=var_col, op=nl.square, data=sum_col, scale=inv_n)  # (sum / N)^2
        nisa.tensor_scalar(dst=sqsum_col, data=sqsum_col, op0=nl.multiply, operand0=inv_n)

        nisa.activation(dst=mean_col, op=nl.copy, data=sum_col, scale=inv_n)
        nisa.tensor_tensor(dst=var_col, data1=sqsum_col, data2=var_col, op=nl.subtract)

        nisa.dma_copy(
            dst=means.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1),
            src=mean_col[:c_out_tile_size, 0],
        )
        nisa.dma_copy(
            dst=variances.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1),
            src=var_col[:c_out_tile_size, 0],
        )


def _rescale_phase(
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    conv_out: nl.NkiTensor,
    y_out: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    batch_norm_eps: float,
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    gamma_buf: nl.NkiTensor,
    beta_buf: nl.NkiTensor,
    true_gamma_buf: nl.NkiTensor,
    neg_true_beta_buf: nl.NkiTensor,
    reload_buf: nl.NkiTensor,
    identity_mask: list,
    shard_stats_on_dh: bool,
    n_prgs: int,
    reload_interleave: int,
    reload_free_tile: int,
    total_spatial: int,
) -> None:
    """
    Rescale phase: precompute per-channel affine coefficients from the SBUF-resident stats,
        true_gamma    = gamma * rsqrt(variances + batch_norm_eps)
        neg_true_beta = means * true_gamma - beta
    and rescale each reloaded conv_out tile with one tensor_scalar
        y_out = conv_out * true_gamma - neg_true_beta
    (== gamma * (conv_out - mean) / sqrt(var + eps) + beta). See conv3d_design_spec.md.

    Each core rescales ONLY the conv_out region it produced itself (own-slice sharding), so no
    core_barrier is needed and the reload overlaps the main loop's tail. A batch-sharded
    alternative would give larger DMAs but must read the peer's conv_out, forcing a barrier that
    serializes the whole finalize behind the whole main loop — a net loss (profiled).
    """
    P_MAX = tile_cfg.P_MAX
    C_OUT_REP = tile_cfg.C_OUT_REP

    # Own-slice sharding: all batches; only the C_out / spatial extent differs (shard_on_dh ->
    # own D-H slice; C_out-shard -> own C_out slice; else all). dh flattens to dh * W_out + w.
    batch_lo, batch_hi = 0, cfg.B
    if shard_stats_on_dh:
        cout_lo, cout_hi = 0, cfg.C_out
        sp_lo = tile_cfg.dh_start * cfg.W_out
        sp_hi = tile_cfg.dh_end * cfg.W_out
    elif n_prgs > 1:
        cout_lo, cout_hi = tile_cfg.C_out_start, tile_cfg.C_out_end
        sp_lo, sp_hi = 0, total_spatial
    else:
        cout_lo, cout_hi = 0, cfg.C_out
        sp_lo, sp_hi = 0, total_spatial

    for c_out_tile_start in range(cout_lo, cout_hi, P_MAX):
        c_out_tile_idx = c_out_tile_start // P_MAX
        c_out_tile_end = min(c_out_tile_start + P_MAX, cout_hi)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        # Partition span of the C_OUT_REP replication (band g at [g * c_out_tile_size, +)).
        rep_active_part = C_OUT_REP * c_out_tile_size

        # Stats read from SBUF columns (no reload); affine computed on band 0, replicated below.
        mean_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        var_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        gamma_col = gamma_buf[:c_out_tile_size, 0:1]
        beta_col = beta_buf[:c_out_tile_size, 0:1]
        true_gamma_base = true_gamma_buf[:c_out_tile_size, 0:1]
        neg_true_beta_base = neg_true_beta_buf[:c_out_tile_size, 0:1]

        nisa.dma_copy(dst=gamma_col, src=gamma.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))
        nisa.dma_copy(dst=beta_col, src=beta.slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1))

        # true_gamma = gamma * rsqrt(variances + eps)
        nisa.activation(dst=true_gamma_base, op=nl.rsqrt, data=var_col, bias=batch_norm_eps)
        nisa.tensor_tensor(dst=true_gamma_base, data1=true_gamma_base, data2=gamma_col, op=nl.multiply)
        # neg_true_beta = means * true_gamma - beta
        nisa.tensor_scalar(
            dst=neg_true_beta_base,
            data=mean_col,
            op0=nl.multiply,
            operand0=true_gamma_base,
            op1=nl.subtract,
            operand1=beta_col,
        )

        # Replicate coefficients into bands >= 1 via nc_stream_shuffle (dst_start quadrant-aligned).
        for band_idx in range(1, C_OUT_REP):
            dst_lo = band_idx * c_out_tile_size
            nisa.nc_stream_shuffle(
                dst=true_gamma_buf[dst_lo : dst_lo + c_out_tile_size, 0:1],
                src=true_gamma_base,
                shuffle_mask=identity_mask,
            )
            nisa.nc_stream_shuffle(
                dst=neg_true_beta_buf[dst_lo : dst_lo + c_out_tile_size, 0:1],
                src=neg_true_beta_base,
                shuffle_mask=identity_mask,
            )

        # Coefficient views spanning all replicated bands (tensor_scalar operands).
        true_gamma = true_gamma_buf[:rep_active_part, 0:1]
        neg_true_beta = neg_true_beta_buf[:rep_active_part, 0:1]

        # Reload conv_out and rescale into y_out. Column tiling handles C_OUT_REP spatial chunks per
        # iteration (rep_step), rep == 1 one tile; multibuffering rotates reload_buf for overlap.
        reload_slot_counter = 0
        rep_step = C_OUT_REP * reload_free_tile
        # Per-chunk plan (band_extents, max_sp_size) is batch-independent, so build it once.
        # band_extents = (band_idx, sp_start, sp_end, sp_size) per active band (tail bands dropped).
        chunk_plan = []
        for chunk_start in range(sp_lo, sp_hi, rep_step):
            band_extents = []
            max_sp_size = 0
            for band_idx in range(C_OUT_REP):
                band_sp_start = chunk_start + band_idx * reload_free_tile
                if band_sp_start >= sp_hi:
                    break
                band_sp_end = min(band_sp_start + reload_free_tile, sp_hi)
                band_sp_size = band_sp_end - band_sp_start
                band_extents.append((band_idx, band_sp_start, band_sp_end, band_sp_size))
                if band_sp_size > max_sp_size:
                    max_sp_size = band_sp_size
            chunk_plan.append((band_extents, max_sp_size))

        for batch_idx in range(batch_lo, batch_hi):
            conv_out_flat = (
                conv_out.select(dim=0, index=batch_idx)
                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(1, 3)
            )
            y_out_flat = (
                y_out.select(dim=0, index=batch_idx)
                .slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
                .flatten_dims(1, 3)
            )
            for plan_entry in chunk_plan:
                band_extents = plan_entry[0]
                max_sp_size = plan_entry[1]
                slot_idx = reload_slot_counter % reload_interleave
                slot_start = slot_idx * reload_free_tile

                # DMA each active band's spatial slice into its partition band.
                for band_ext in band_extents:
                    band_idx = band_ext[0]
                    band_sp_start = band_ext[1]
                    band_sp_end = band_ext[2]
                    band_sp_size = band_ext[3]
                    band_part_lo = band_idx * c_out_tile_size
                    conv_out_view = conv_out_flat.slice(dim=1, start=band_sp_start, end=band_sp_end, step=1)
                    band_load = reload_buf[
                        band_part_lo : band_part_lo + c_out_tile_size,
                        slot_start : slot_start + band_sp_size,
                    ]
                    nisa.dma_copy(dst=band_load, src=conv_out_view)

                # One tensor_scalar rescales all active bands with the replicated coeffs. A short
                # band's [band_sp_size:max_sp_size) tail is garbage but never stored.
                active_bands = len(band_extents)
                active_part = active_bands * c_out_tile_size
                compute_tile = reload_buf[:active_part, slot_start : slot_start + max_sp_size]
                nisa.tensor_scalar(
                    dst=compute_tile,
                    data=compute_tile,
                    op0=nl.multiply,
                    operand0=true_gamma[:active_part, 0:1],
                    op1=nl.subtract,
                    operand1=neg_true_beta[:active_part, 0:1],
                )
                # Activation (post-batchnorm) on the normalized tile.
                if cfg.has_activation:
                    nisa.activation(dst=compute_tile, data=compute_tile, op=get_nl_act_fn_from_type(cfg.activation_fn))

                # DMA each active band's rescaled slice back to its y_out region.
                for band_ext in band_extents:
                    band_idx = band_ext[0]
                    band_sp_start = band_ext[1]
                    band_sp_end = band_ext[2]
                    band_sp_size = band_ext[3]
                    band_part_lo = band_idx * c_out_tile_size
                    y_out_view = y_out_flat.slice(dim=1, start=band_sp_start, end=band_sp_end, step=1)
                    band_store = reload_buf[
                        band_part_lo : band_part_lo + c_out_tile_size,
                        slot_start : slot_start + band_sp_size,
                    ]
                    nisa.dma_copy(dst=y_out_view, src=band_store)
                reload_slot_counter += 1


def _momentum_update_phase(
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    means_sbuf: nl.NkiTensor,
    variances_sbuf: nl.NkiTensor,
    running_means: nl.NkiTensor,
    running_variances: nl.NkiTensor,
    updated_running_means: nl.NkiTensor,
    updated_running_variances: nl.NkiTensor,
    run_running_mean_buf: nl.NkiTensor,
    run_new_mean_buf: nl.NkiTensor,
    run_running_var_buf: nl.NkiTensor,
    run_new_var_buf: nl.NkiTensor,
    momentum: float,
    shard_stats_on_dh: bool,
) -> None:
    """
    Momentum-update phase (matches torch.nn.BatchNorm2d), sharded by C_out slice so each core
    only touches the stats it produced:
        updated_running_means     = (1 - momentum) * running_means     + momentum * means
        updated_running_variances = (1 - momentum) * running_variances + momentum * (N/(N-1)) * var
    i.e. the running variance uses the correction-1 (unbiased) variance. Reads the fresh mean /
    variance from the SBUF columns (no HBM reload).
    """
    P_MAX = tile_cfg.P_MAX
    one_minus_momentum = 1.0 - momentum
    N = cfg.B * cfg.D_out * cfg.H_out * cfg.W_out
    correct1_factor = N / (N - 1) if N > 1 else 1.0

    if shard_stats_on_dh:
        upd_c_out_start, upd_c_out_end = 0, cfg.C_out
    else:
        upd_c_out_start, upd_c_out_end = tile_cfg.C_out_start, tile_cfg.C_out_end

    for c_out_tile_start in range(upd_c_out_start, upd_c_out_end, P_MAX):
        c_out_tile_idx = c_out_tile_start // P_MAX
        c_out_tile_end = min(c_out_tile_start + P_MAX, upd_c_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        running_mean_col = run_running_mean_buf[:c_out_tile_size, 0:1]
        new_mean_col = run_new_mean_buf[:c_out_tile_size, 0:1]
        running_var_col = run_running_var_buf[:c_out_tile_size, 0:1]
        new_var_col = run_new_var_buf[:c_out_tile_size, 0:1]
        # Fresh stats read from the SBUF columns (no HBM reload).
        mean_sbuf_col = means_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]
        var_sbuf_col = variances_sbuf[:c_out_tile_size, c_out_tile_idx : c_out_tile_idx + 1]

        # Mean: new_scale = 1; variance: correction-1 factor. Disjoint scratch avoids serializing.
        _momentum_update_stat(
            running_hbm=running_means,
            new_stat_col=mean_sbuf_col,
            updated_hbm=updated_running_means,
            running_scratch=running_mean_col,
            new_scratch=new_mean_col,
            c_out_tile_start=c_out_tile_start,
            c_out_tile_end=c_out_tile_end,
            one_minus_momentum=one_minus_momentum,
            momentum=momentum,
            new_scale=1.0,
        )
        _momentum_update_stat(
            running_hbm=running_variances,
            new_stat_col=var_sbuf_col,
            updated_hbm=updated_running_variances,
            running_scratch=running_var_col,
            new_scratch=new_var_col,
            c_out_tile_start=c_out_tile_start,
            c_out_tile_end=c_out_tile_end,
            one_minus_momentum=one_minus_momentum,
            momentum=momentum,
            new_scale=correct1_factor,
        )


def _finalize_batch_norm(
    cfg: "Conv3dConfig",
    tile_cfg: "Conv3dTileConfig",
    sbm: SbufManager,
    conv_out: nl.NkiTensor,
    y_out: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: nl.NkiTensor,
    batch_norm_eps: float,
    bn_sum_bufs: nl.NkiTensor,
    bn_sqsum_bufs: nl.NkiTensor,
    bn_sqsum_partials: nl.NkiTensor,
    means: nl.NkiTensor,
    variances: nl.NkiTensor,
    n_prgs: int,
    prg_id: int,
    momentum: float,
    running_means: nl.NkiTensor,
    running_variances: nl.NkiTensor,
    updated_running_means: nl.NkiTensor,
    updated_running_variances: nl.NkiTensor,
    bn_reload_interleave: int,
) -> None:
    """
    Finalize fused batchnorm: allocate the shared finalize SBUF buffers, then run the three
    phases (each its own function, see their docstrings): _finalize_stats_phase (mean / variance),
    _rescale_phase (normalize conv_out -> y_out + optional activation), _momentum_update_phase
    (running stats). Each core only reads back the region it produced itself, so no cross-core HBM
    read / core_barrier is needed. See conv3d_design_spec.md.
    """
    P_MAX = tile_cfg.P_MAX
    full_c_out_tile_count = div_ceil(cfg.C_out, P_MAX)
    tile_size_full = min(P_MAX, cfg.C_out)
    inv_n = 1.0 / (cfg.B * cfg.D_out * cfg.H_out * cfg.W_out)
    total_spatial = cfg.D_out * cfg.H_out * cfg.W_out
    reload_free_tile = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    shard_stats_on_dh = tile_cfg.shard_on_dh and n_prgs > 1

    # Receive buffers for the cross-core sum(Y)/sum(Y^2) sendrecv (shard_on_dh path).
    recv_sum = sbm.alloc_heap(
        shape=(tile_cfg.c_out_tile_size_max, full_c_out_tile_count), dtype=nl.float32, name="bn_recv_sum"
    )
    recv_sqsum = sbm.alloc_heap(
        shape=(tile_cfg.c_out_tile_size_max, full_c_out_tile_count), dtype=nl.float32, name="bn_recv_sqsum"
    )
    # Persistent final mean / variance, one column per global C_out tile: stats phase writes here
    # (+ HBM), later phases read without reloading.
    means_sbuf = sbm.alloc_heap(shape=(tile_size_full, full_c_out_tile_count), dtype=nl.float32, name="bn_means_sbuf")
    variances_sbuf = sbm.alloc_heap(
        shape=(tile_size_full, full_c_out_tile_count), dtype=nl.float32, name="bn_variances_sbuf"
    )
    # Rescale-phase affine scratch. Under column tiling true_gamma / neg_true_beta (and reload_buf)
    # are C_OUT_REP * tile_size_full tall so one tensor_scalar rescales all bands at once.
    C_OUT_REP = tile_cfg.C_OUT_REP
    rep_part_dim = C_OUT_REP * tile_size_full
    gamma_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_gamma")
    beta_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_beta")
    true_gamma_buf = sbm.alloc_heap(shape=(rep_part_dim, 1), dtype=nl.float32, name="bn_true_gamma")
    neg_true_beta_buf = sbm.alloc_heap(shape=(rep_part_dim, 1), dtype=nl.float32, name="bn_neg_true_beta")
    # Momentum-phase scratch: separate mean / variance buffers so the two updates don't alias.
    run_running_mean_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_running_mean")
    run_new_mean_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_new_mean")
    run_running_var_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_running_var")
    run_new_var_buf = sbm.alloc_heap(shape=(tile_size_full, 1), dtype=nl.float32, name="bn_run_new_var")

    # Multibuffered reload buffer, allocated LAST (popped first) so its interleave is sized from free
    # SBUF, capped at bn_reload_interleave. First reserve for the stats phase's gather buffers.
    reload_slot_bytes = reload_free_tile * sizeinbytes(y_out.dtype)
    reload_slot_cost = reload_slot_bytes + _heap_align_waste(reload_slot_bytes)
    reserve_for_gather = 0
    if C_OUT_REP > 1:
        gather_bytes = C_OUT_REP * sizeinbytes(nl.float32)
        reserve_for_gather = 2 * (gather_bytes + _heap_align_waste(gather_bytes))
    free_slots = max(1, (sbm.get_free_space() - reserve_for_gather) // reload_slot_cost)
    reload_interleave = min(bn_reload_interleave, free_slots)
    reload_buf = sbm.alloc_heap(
        shape=(rep_part_dim, reload_interleave * reload_free_tile), dtype=y_out.dtype, name="bn_reload"
    )

    # Identity shuffle mask (partition i <- i per 32-partition quadrant), shared by both
    # column-tiling steps. Plain list — the tracer rejects tuple(range(...)).
    identity_mask = []
    for _mask_i in range(_PARTITION_STRIDE_32):
        identity_mask.append(_mask_i)

    # --- Finalize-stats phase: means / variances (HBM + persistent SBUF columns) ---
    _finalize_stats_phase(
        cfg=cfg,
        tile_cfg=tile_cfg,
        sbm=sbm,
        bn_sum_bufs=bn_sum_bufs,
        bn_sqsum_bufs=bn_sqsum_bufs,
        bn_sqsum_partials=bn_sqsum_partials,
        recv_sum=recv_sum,
        recv_sqsum=recv_sqsum,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        means=means,
        variances=variances,
        identity_mask=identity_mask,
        shard_stats_on_dh=shard_stats_on_dh,
        prg_id=prg_id,
        inv_n=inv_n,
    )

    # --- Rescale phase: rescale conv_out into y_out using the SBUF-resident statistics ---
    _rescale_phase(
        cfg=cfg,
        tile_cfg=tile_cfg,
        conv_out=conv_out,
        y_out=y_out,
        gamma=gamma,
        beta=beta,
        batch_norm_eps=batch_norm_eps,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        gamma_buf=gamma_buf,
        beta_buf=beta_buf,
        true_gamma_buf=true_gamma_buf,
        neg_true_beta_buf=neg_true_beta_buf,
        reload_buf=reload_buf,
        identity_mask=identity_mask,
        shard_stats_on_dh=shard_stats_on_dh,
        n_prgs=n_prgs,
        reload_interleave=reload_interleave,
        reload_free_tile=reload_free_tile,
        total_spatial=total_spatial,
    )

    # --- Momentum-update phase: update the running statistics ---
    _momentum_update_phase(
        cfg=cfg,
        tile_cfg=tile_cfg,
        means_sbuf=means_sbuf,
        variances_sbuf=variances_sbuf,
        running_means=running_means,
        running_variances=running_variances,
        updated_running_means=updated_running_means,
        updated_running_variances=updated_running_variances,
        run_running_mean_buf=run_running_mean_buf,
        run_new_mean_buf=run_new_mean_buf,
        run_running_var_buf=run_running_var_buf,
        run_new_var_buf=run_new_var_buf,
        momentum=momentum,
        shard_stats_on_dh=shard_stats_on_dh,
    )

    # Free scratch (allocated last, popped first).
    sbm.pop_heap()  # bn_reload
    sbm.pop_heap()  # bn_run_new_var
    sbm.pop_heap()  # bn_run_running_var
    sbm.pop_heap()  # bn_run_new_mean
    sbm.pop_heap()  # bn_run_running_mean
    sbm.pop_heap()  # bn_neg_true_beta
    sbm.pop_heap()  # bn_true_gamma
    sbm.pop_heap()  # bn_beta
    sbm.pop_heap()  # bn_gamma
    sbm.pop_heap()  # bn_variances_sbuf
    sbm.pop_heap()  # bn_means_sbuf
    sbm.pop_heap()  # bn_recv_sqsum
    sbm.pop_heap()  # bn_recv_sum


def _heap_align_waste(bytes_per_partition: int) -> int:
    """Compute alignment waste for a single SBM heap allocation."""
    return (_HEAP_ALIGN - (bytes_per_partition % _HEAP_ALIGN)) % _HEAP_ALIGN


def _decompose_k_position(k_position: int, K_h: int, K_w: int) -> tuple[int, int, int]:
    """Decompose a flat filter position index into (k_d, k_h, k_w) indices."""
    k_d_idx, k_hw_remainder = divmod(k_position, K_h * K_w)
    k_h_idx, k_w_idx = divmod(k_hw_remainder, K_w)
    return k_d_idx, k_h_idx, k_w_idx


def _compute_used_k_positions(cfg: "Conv3dConfig") -> list[int]:
    """
    Compute which flat filter positions are actually used.

    For each filter position (k_d, k_h, k_w), checks whether there exists any
    output position where the corresponding input position falls within valid
    (non-padded) bounds.

    Args:
        cfg: Conv3dConfig with all convolution parameters.

    Returns:
        List of flat filter position indices (into K_d*K_h*K_w) that are actually used.
    """
    K_d, K_h, K_w = cfg.K_d, cfg.K_h, cfg.K_w

    used_k_d = [False] * K_d
    for k_d in range(K_d):
        for d_out in range(cfg.D_out):
            d_in = d_out * cfg.stride_d + k_d * cfg.dilation_d - cfg.pad_d_left
            if 0 <= d_in < cfg.D:
                used_k_d[k_d] = True
                break

    used_k_h = [False] * K_h
    for k_h in range(K_h):
        for h_out in range(cfg.H_out):
            h_in = h_out * cfg.stride_h + k_h * cfg.dilation_h - cfg.pad_h_top
            if 0 <= h_in < cfg.H:
                used_k_h[k_h] = True
                break

    used_k_w = [False] * K_w
    for k_w in range(K_w):
        for w_out in range(cfg.W_out):
            w_in = w_out * cfg.stride_w + k_w * cfg.dilation_w - cfg.pad_w_left
            if 0 <= w_in < cfg.W:
                used_k_w[k_w] = True
                break

    used_positions = []
    for k_d in range(K_d):
        for k_h in range(K_h):
            for k_w in range(K_w):
                if used_k_d[k_d] and used_k_h[k_h] and used_k_w[k_w]:
                    flat_idx = k_d * K_h * K_w + k_h * K_w + k_w
                    used_positions.append(flat_idx)

    return used_positions


@dataclass
class Conv3dConfig(nl.NKIObject):
    """Configuration for 3D convolution kernel parameters and input/output dimensions."""

    stride_d: int
    stride_h: int
    stride_w: int
    pad_d_left: int
    pad_d_right: int
    pad_h_top: int
    pad_h_bottom: int
    pad_w_left: int
    pad_w_right: int
    dilation_d: int
    dilation_h: int
    dilation_w: int
    has_bias: bool
    has_activation: bool
    activation_fn: Optional[ActFnType]
    lnc_shard: bool
    fuse_batch_norm: bool
    B: int
    C_in: int
    C_out: int
    D: int
    H: int
    W: int
    K_d: int
    K_h: int
    K_w: int
    D_out: int
    H_out: int
    W_out: int


@dataclass
class Conv3dTileConfig(nl.NKIObject):
    """Tiling configuration including partition sizes, tile counts, and sharding ranges."""

    P_MAX: int
    F_MAX: int
    PSUM_BANK_SIZE: int
    NUM_PSUM_BANKS: int
    W_tile: int
    c_out_tile_size_max: int
    c_in_tile_size_max: int
    c_in_tile_count: int
    c_out_tile_count: int
    K_outer_tile_count: int
    K_REP_max: int
    partition_stride_max: int
    stacked_filter_dim_max: int
    C_out_start: int
    C_out_end: int
    C_out_local: int
    num_dh_stacked: int
    # Column tiling: C_OUT_REP bands (1 disables), PE column-tile size col_size, per-band D-H count
    # dh_head_size. Band g runs on column tile / PSUM band g * col_size. See conv3d_design_spec.md.
    C_OUT_REP: int
    col_size: int
    dh_head_size: int
    d_window_max: int
    h_window_max: int
    w_window_max: int
    shard_on_dh: bool
    dh_start: int
    dh_end: int
    used_k_positions: list[int]
    K_total_used: int
    h_chunk_size: int
    # Eviction flushes per C_out tile (B * num_dh_groups * num_w_tiles); sizes the sum(Y^2)
    # partials buffer (one column each).
    bn_partial_iters: int
    # TODO: Determine optimal engine offload values using a cost model.
    tensor_copy_activation_engine_offload: int
    memset_gpsimd_engine_offload: int


@dataclass
class Conv3dMemoryConfig(nl.NKIObject):
    """Memory layout configuration for SBUF buffer allocation and interleaving."""

    dtype_size: int
    TOTAL_SBUF: int
    bias_size_per_tile: int
    filters_per_c_out_tile: int
    per_c_out_tile_outer: int
    per_c_out_tile_inner: int
    input_window_memory: int
    stacked_input_memory: int
    c_out_interleave: int
    input_window_interleave: int
    stacked_input_interleave: int
    w_out_interleave: int
    num_w_out_tiles: int
    total_memory_required: int
    # Store-batch merging (column tiling): store_batch_merge batches per band DMA, store_pipe slots
    # overlap store with compute; product == w_out_interleave (== 1 disables). See _build_memory_config.
    store_batch_merge: int
    store_pipe: int
    # Upper bound on rescale-phase reload multibuffering slots; _finalize_batch_norm picks the real
    # interleave by fitting up to this many into the free heap. Set in _build_memory_config.
    bn_reload_interleave: int


def _build_conv3d_config(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
    dilation: tuple[int, int, int],
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
    fuse_batch_norm: bool,
) -> Conv3dConfig:
    """
    Build Conv3dConfig from input tensors and convolution parameters.

    Extracts dimensions from input/filter shapes and computes output spatial
    dimensions based on stride, padding, and dilation settings.
    """
    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation
    B, C_in, D, H, W = x_in.shape
    K_d, K_h, K_w, _, C_out = filters.shape
    D_out = (D + pad_d_left + pad_d_right - dilation_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H + pad_h_top + pad_h_bottom - dilation_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + pad_w_left + pad_w_right - dilation_w * (K_w - 1) - 1) // stride_w + 1
    return Conv3dConfig(
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_d_left=pad_d_left,
        pad_d_right=pad_d_right,
        pad_h_top=pad_h_top,
        pad_h_bottom=pad_h_bottom,
        pad_w_left=pad_w_left,
        pad_w_right=pad_w_right,
        dilation_d=dilation_d,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        has_bias=bias != None,
        has_activation=activation_fn != None,
        activation_fn=activation_fn,
        lnc_shard=lnc_shard,
        fuse_batch_norm=fuse_batch_norm,
        B=B,
        C_in=C_in,
        C_out=C_out,
        D=D,
        H=H,
        W=W,
        K_d=K_d,
        K_h=K_h,
        K_w=K_w,
        D_out=D_out,
        H_out=H_out,
        W_out=W_out,
    )


def _get_k_replication_params(c_in_tile_size: int, K_total: int) -> tuple[int, int]:
    """Determine K-replication count and partition stride based on input channel tile size."""
    if c_in_tile_size <= _PARTITION_STRIDE_32:
        return min(K_total, _MAX_K_REP_AT_32), _PARTITION_STRIDE_32
    elif c_in_tile_size <= _PARTITION_STRIDE_64:
        return min(K_total, _MAX_K_REP_AT_64), _PARTITION_STRIDE_64
    else:
        return 1, c_in_tile_size


def _get_c_out_replication_params(c_out_tile_size: int) -> tuple[int, int]:
    """
    Determine the C_out column-tiling replication factor and PE column-tile size.

    When the C_out tile size is smaller than the 128 PE columns, the nc_matmul
    stationary free dim (= C_out) occupies only that many columns and leaves the rest
    of the systolic array idle. Column tiling packs C_OUT_REP D-H groups into a single
    output flush, each running its matmuls on a different PE column tile (start column
    g * col_size) and landing in a different partition band of one wide PSUM bank. Each
    group keeps the full baseline free dim, so the total matmul count and per-matmul free
    dim are unchanged relative to processing the groups one-flush-at-a-time; the win is
    parallel PE-column occupancy and a single wide eviction.

    C_OUT_REP is bounded by the column-tile size that fits C_out (4x32 or 2x64 columns)
    and the hardware (4x32 needs NeuronCore-v3+; 2x64 needs v2+).

    Returns (C_OUT_REP, col_size). C_OUT_REP == 1 means no column tiling (col_size 128).
    """
    nc_ver = nisa.get_nc_version()
    if c_out_tile_size <= _COLUMN_TILE_32 and nc_ver >= nisa.nc_version.gen3:
        return _MAX_K_REP_AT_32, _COLUMN_TILE_32
    elif c_out_tile_size <= _COLUMN_TILE_64:
        return _MAX_K_REP_AT_64, _COLUMN_TILE_64
    else:
        return 1, nl.tile_size.pmax


def _count_dh_groups(dh_start: int, dh_end: int, num_dh_stacked: int, H_out: int) -> int:
    """
    Count the D-H output groups the main loop iterates over for a given dh range.

    Mirrors the grouping in the conv3d loop nest: starting at dh_start, each group packs
    up to num_dh_stacked consecutive flat (d_out, h_out) positions but never crosses a
    d_out boundary. Used to size the batchnorm squared-sum partials buffer exactly.
    """
    count = 0
    dh_group_idx = dh_start
    while dh_group_idx < dh_end:
        first_d_out = None
        group_size = 0
        for dh_stacked_idx in range(num_dh_stacked):
            flat_idx = dh_group_idx + dh_stacked_idx
            if flat_idx >= dh_end:
                break
            d_out_idx = flat_idx // H_out
            if first_d_out == None:
                first_d_out = d_out_idx
            elif d_out_idx != first_d_out:
                break
            group_size += 1
        count += 1
        dh_group_idx += group_size
    return count


def _calculate_union_window_size(cfg: Conv3dConfig, num_dh_stacked: int, w_tile: int) -> tuple[int, int, int]:
    """Calculate the union input window size for a group of stacked D-H output positions."""
    single_d_window = (cfg.K_d - 1) * cfg.dilation_d + 1
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    w_window = (w_tile - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w + 1
    if num_dh_stacked == 1:
        return single_d_window, single_h_window, w_window
    max_d_out_in_group = (num_dh_stacked - 1 + cfg.H_out - 1) // cfg.H_out
    d_window_max = min(single_d_window + max_d_out_in_group * cfg.stride_d, cfg.D)
    max_h_out_in_group = min(num_dh_stacked - 1, cfg.H_out - 1)
    h_window_max = min(single_h_window + max_h_out_in_group * cfg.stride_h, cfg.H)
    return d_window_max, h_window_max, w_window


def _build_tile_config(
    cfg: Conv3dConfig,
    dtype_size: int,
    total_sbuf_budget: Optional[int] = None,
) -> Conv3dTileConfig:
    """Build tiling configuration with LNC sharding and iterative memory fitting."""
    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    TOTAL_SBUF = total_sbuf_budget if total_sbuf_budget != None else nl.tile_size.total_available_sbuf_size
    _, n_prgs, prg_id = get_verified_program_sharding_info("conv3d", (0, 1))

    full_c_out_tile_count = div_ceil(cfg.C_out, P_MAX)
    total_dh_positions = cfg.D_out * cfg.H_out

    dh_work_per_nc = div_ceil(total_dh_positions, 2)
    cout_tiles_per_nc = div_ceil(full_c_out_tile_count, 2)
    dh_waste = 2 * dh_work_per_nc - total_dh_positions
    cout_waste = 2 * cout_tiles_per_nc - full_c_out_tile_count
    shard_on_dh = (total_dh_positions >= 2) and (dh_waste <= cout_waste)

    if n_prgs > 1 and shard_on_dh:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_per_nc = div_ceil(total_dh_positions, n_prgs)
        dh_start = dh_per_nc * prg_id
        dh_end = min(dh_start + dh_per_nc, total_dh_positions)
    elif n_prgs > 1:
        C_out_per_nc = div_ceil(cfg.C_out, n_prgs)
        C_out_start = C_out_per_nc * prg_id
        C_out_end = min(C_out_start + C_out_per_nc, cfg.C_out)
        C_out_local = C_out_end - C_out_start
        dh_start, dh_end = 0, total_dh_positions
    else:
        C_out_start, C_out_end, C_out_local = 0, cfg.C_out, cfg.C_out
        dh_start, dh_end = 0, total_dh_positions
        shard_on_dh = False

    c_out_tile_size_max = min(P_MAX, C_out_local)
    c_in_tile_size_max = min(P_MAX, cfg.C_in)
    used_k_positions = _compute_used_k_positions(cfg)
    K_total_used = len(used_k_positions)
    K_REP_max, partition_stride_max = _get_k_replication_params(c_in_tile_size_max, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP_max)
    stacked_filter_dim_max = partition_stride_max * K_REP_max
    c_in_tile_count = div_ceil(cfg.C_in, P_MAX)
    c_out_tile_count = div_ceil(C_out_local, P_MAX)

    # C_out column tiling (low C_out): pack C_OUT_REP D-H bands (dh_head each, num_dh_stacked =
    # C_OUT_REP * dh_head) onto parallel PE columns in one flush. See conv3d_design_spec.md.
    C_OUT_REP, col_size = _get_c_out_replication_params(c_out_tile_size_max)

    W_tile = min(cfg.W_out, F_MAX)
    dh_head = min(F_MAX // cfg.W_out, total_dh_positions) if cfg.W_out < F_MAX and total_dh_positions > 1 else 1

    while True:
        # Stack C_OUT_REP dh_head-wide bands, capped so the group doesn't exceed the total D-H work.
        c_out_rep_eff = min(C_OUT_REP, max(1, total_dh_positions // dh_head)) if C_OUT_REP > 1 else 1
        num_dh_stacked = c_out_rep_eff * dh_head
        per_band_free = dh_head * W_tile
        effective_free_dim = num_dh_stacked * W_tile
        d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)
        bias_mem = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
        filter_mem = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size
        # PSUM/result are per_band_free wide (bands stacked on partitions); stacked input / input
        # window span the whole packed free dim.
        min_mem = (
            bias_mem
            + filter_mem
            + per_band_free * dtype_size
            + d_window_max * h_window_max * w_window_max * dtype_size
            + K_outer_tile_count * effective_free_dim * dtype_size
        )
        if min_mem <= TOTAL_SBUF:
            break
        if dh_head > 1:
            dh_head = max(1, dh_head // 2)
        elif W_tile > 1:
            W_tile = max(1, W_tile // 2)
        else:
            break

    c_out_rep_eff = min(C_OUT_REP, max(1, total_dh_positions // dh_head)) if C_OUT_REP > 1 else 1
    C_OUT_REP = c_out_rep_eff
    if C_OUT_REP == 1:
        col_size = P_MAX
    num_dh_stacked = C_OUT_REP * dh_head
    dh_head_size = dh_head
    d_window_max, h_window_max, w_window_max = _calculate_union_window_size(cfg, num_dh_stacked, W_tile)

    # H-chunk size: input reuse across consecutive h_out positions.
    single_h_window = (cfg.K_h - 1) * cfg.dilation_h + 1
    num_w_tiles = div_ceil(cfg.W_out, W_tile)
    if num_dh_stacked == 1 and cfg.K_h > 1 and cfg.H_out > 1 and num_w_tiles == 1:
        h_chunk_size = cfg.K_h

        h_chunk_h_window = single_h_window + (h_chunk_size - 1) * cfg.stride_h
        h_chunk_h_window = min(h_chunk_h_window, cfg.H)

        effective_free_dim_check = num_dh_stacked * W_tile
        while h_chunk_size > 1:
            h_chunk_h_window = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
            chunk_input_window_mem = d_window_max * h_chunk_h_window * w_window_max * dtype_size
            bias_mem_check = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
            filter_mem_check = c_in_tile_count * K_outer_tile_count * c_out_tile_size_max * dtype_size

            min_mem_check = (
                bias_mem_check
                + filter_mem_check
                + effective_free_dim_check * dtype_size
                + chunk_input_window_mem
                + K_outer_tile_count * effective_free_dim_check * dtype_size
            )
            if min_mem_check <= TOTAL_SBUF:
                break
            h_chunk_size -= 1
        h_chunk_h_window_max = min(single_h_window + (h_chunk_size - 1) * cfg.stride_h, cfg.H)
    else:
        h_chunk_size = 1
        h_chunk_h_window_max = h_window_max

    # Flushes per C_out tile (batches x D-H groups x W tiles), one sum(Y^2) partials column each.
    # A flush is one C_OUT_REP-band group; h-chunking forces C_OUT_REP == 1, so they never mix.
    num_dh_groups = _count_dh_groups(dh_start, dh_end, num_dh_stacked, cfg.H_out)
    bn_partial_iters = cfg.B * num_dh_groups * num_w_tiles

    tile_cfg = Conv3dTileConfig(
        P_MAX=P_MAX,
        F_MAX=F_MAX,
        PSUM_BANK_SIZE=_PSUM_BANK_SIZE,
        NUM_PSUM_BANKS=_NUM_PSUM_BANKS,
        W_tile=W_tile,
        c_out_tile_size_max=c_out_tile_size_max,
        c_in_tile_size_max=c_in_tile_size_max,
        c_in_tile_count=c_in_tile_count,
        c_out_tile_count=c_out_tile_count,
        K_outer_tile_count=K_outer_tile_count,
        K_REP_max=K_REP_max,
        partition_stride_max=partition_stride_max,
        stacked_filter_dim_max=stacked_filter_dim_max,
        C_out_start=C_out_start,
        C_out_end=C_out_end,
        C_out_local=C_out_local,
        num_dh_stacked=num_dh_stacked,
        C_OUT_REP=C_OUT_REP,
        col_size=col_size,
        dh_head_size=dh_head_size,
        d_window_max=d_window_max,
        h_window_max=h_chunk_h_window_max,
        w_window_max=w_window_max,
        shard_on_dh=shard_on_dh,
        dh_start=dh_start,
        dh_end=dh_end,
        used_k_positions=used_k_positions,
        K_total_used=K_total_used,
        h_chunk_size=h_chunk_size,
        bn_partial_iters=bn_partial_iters,
        tensor_copy_activation_engine_offload=4,
        memset_gpsimd_engine_offload=2,
    )

    K_total_full = cfg.K_d * cfg.K_h * cfg.K_w
    logger = get_logger("conv3d")
    logger.debug(
        f"TileConfig: W_tile={tile_cfg.W_tile}, num_dh_stacked={tile_cfg.num_dh_stacked}, "
        f"c_out_tile_size_max={tile_cfg.c_out_tile_size_max}, c_in_tile_size_max={tile_cfg.c_in_tile_size_max}, "
        f"c_in_tile_count={tile_cfg.c_in_tile_count}, c_out_tile_count={tile_cfg.c_out_tile_count}, "
        f"K_outer_tile_count={tile_cfg.K_outer_tile_count}, K_REP_max={tile_cfg.K_REP_max}, "
        f"K_total_used={tile_cfg.K_total_used}/{K_total_full}, "
        f"h_chunk_size={tile_cfg.h_chunk_size}, h_window_max={tile_cfg.h_window_max}, "
        f"shard_on_dh={tile_cfg.shard_on_dh}, dh_range=[{tile_cfg.dh_start},{tile_cfg.dh_end}), "
        f"C_out_range=[{tile_cfg.C_out_start},{tile_cfg.C_out_end}), "
        f"C_OUT_REP={tile_cfg.C_OUT_REP}, col_size={tile_cfg.col_size}, dh_head_size={tile_cfg.dh_head_size}"
    )

    return tile_cfg


def _build_memory_config(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    dtype_size: int,
    total_sbuf_budget: Optional[int] = None,
) -> Conv3dMemoryConfig:
    """Build memory configuration for SBUF buffer allocation and interleaving."""
    base_budget = total_sbuf_budget if total_sbuf_budget != None else nl.tile_size.total_available_sbuf_size
    TOTAL_SBUF = base_budget - _SBUF_RESERVE
    num_w_out_tiles = div_ceil(cfg.W_out, tile_cfg.W_tile)
    bias_size_per_tile = _BIAS_DTYPE_SIZE if cfg.has_bias else 0
    filters_per_c_out_tile = (
        tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * tile_cfg.c_out_tile_size_max * dtype_size
    )
    per_c_out_tile_outer = bias_size_per_tile + filters_per_c_out_tile
    effective_free_dim = tile_cfg.num_dh_stacked * tile_cfg.W_tile
    # Result/PSUM free footprint is one band's width (bands stack on partitions); stacked-input /
    # input-window span the full packed free dim. result_free_dim == effective_free_dim at rep 1.
    result_free_dim = tile_cfg.dh_head_size * tile_cfg.W_tile
    per_c_out_tile_inner = result_free_dim * dtype_size
    input_window_memory = tile_cfg.d_window_max * tile_cfg.h_window_max * tile_cfg.w_window_max * dtype_size
    stacked_input_memory = tile_cfg.K_outer_tile_count * effective_free_dim * dtype_size
    bias_align_waste = _heap_align_waste(_BIAS_DTYPE_SIZE) if cfg.has_bias else 0
    result_align_waste = _heap_align_waste(result_free_dim * dtype_size)
    input_window_align_waste = _heap_align_waste(input_window_memory)
    stacked_input_align_waste = _heap_align_waste(effective_free_dim * dtype_size)
    input_window_cost = input_window_memory + input_window_align_waste
    stacked_input_cost = stacked_input_memory + tile_cfg.K_outer_tile_count * stacked_input_align_waste

    # Reserve a fixed SBUF amount for the batchnorm accumulators; the fit search uses the rest.
    bn_memory = 0
    if cfg.fuse_batch_norm:
        full_c_out_tile_count = div_ceil(cfg.C_out, tile_cfg.P_MAX)
        acc_bytes = full_c_out_tile_count * _BIAS_DTYPE_SIZE
        # bn_sq_scratch mirrors the result tile (one band wide per partition).
        sq_scratch_bytes = result_free_dim * _BIAS_DTYPE_SIZE
        # Per-flush sum(Y^2) partials: one fp32 column per (C_out tile, output-tile flush).
        partials_bytes = full_c_out_tile_count * tile_cfg.bn_partial_iters * _BIAS_DTYPE_SIZE
        bn_memory = (
            2 * (acc_bytes + _heap_align_waste(acc_bytes))
            + (sq_scratch_bytes + _heap_align_waste(sq_scratch_bytes))
            + (partials_bytes + _heap_align_waste(partials_bytes))
        )
    budget = TOTAL_SBUF - bn_memory

    c_out_interleave = 1
    w_out_interleave = _NUM_PSUM_BANKS
    input_window_interleave = 2
    stacked_input_interleave = 2

    found_fit = False
    for min_buf_count in (2, 1):
        if found_fit:
            break
        for try_c_out in range(tile_cfg.c_out_tile_count, 0, -1):
            if found_fit:
                break
            c_out_wide_try = try_c_out * tile_cfg.c_out_tile_size_max
            filter_align_waste_try = _heap_align_waste(c_out_wide_try * dtype_size)
            fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste_try
            outer_cost = try_c_out * per_c_out_tile_outer + try_c_out * bias_align_waste + fixed_filter_align
            for try_w_out in range(_NUM_PSUM_BANKS, 0, -1):
                inner_cost = try_w_out * try_c_out * (per_c_out_tile_inner + result_align_waste)
                min_total = (
                    outer_cost + inner_cost + min_buf_count * input_window_cost + min_buf_count * stacked_input_cost
                )
                if min_total <= budget:
                    c_out_interleave, w_out_interleave = try_c_out, try_w_out
                    input_window_interleave = min_buf_count
                    stacked_input_interleave = min_buf_count
                    found_fit = True
                    break

    kernel_assert(
        found_fit,
        f"Failed to find an SBUF-fitting memory configuration for conv3d. "
        f"TOTAL_SBUF={TOTAL_SBUF}, per_c_out_tile_outer={per_c_out_tile_outer}, "
        f"per_c_out_tile_inner={per_c_out_tile_inner}, "
        f"input_window_cost={input_window_cost}, stacked_input_cost={stacked_input_cost}, "
        f"c_out_tile_count={tile_cfg.c_out_tile_count}",
    )

    c_out_wide = c_out_interleave * tile_cfg.c_out_tile_size_max
    filter_align_waste = _heap_align_waste(c_out_wide * dtype_size)
    fixed_filter_align = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * filter_align_waste
    baseline = (
        c_out_interleave * per_c_out_tile_outer
        + c_out_interleave * bias_align_waste
        + fixed_filter_align
        + w_out_interleave * c_out_interleave * (per_c_out_tile_inner + result_align_waste)
        + input_window_interleave * input_window_cost
        + stacked_input_interleave * stacked_input_cost
    )
    remaining = budget - baseline

    both_cost = input_window_cost + stacked_input_cost
    while remaining >= both_cost and both_cost > 0:
        input_window_interleave += 1
        stacked_input_interleave += 1
        remaining -= both_cost
    if remaining >= stacked_input_cost and stacked_input_cost > 0:
        stacked_input_interleave += 1
        remaining -= stacked_input_cost
    elif remaining >= input_window_cost and input_window_cost > 0:
        input_window_interleave += 1
        remaining -= input_window_cost

    # Account for SBM heap alignment overhead
    num_bias_allocs = c_out_interleave if cfg.has_bias else 0
    num_filter_allocs = tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count
    num_result_allocs = w_out_interleave * c_out_interleave
    num_input_window_allocs = input_window_interleave
    num_stacked_input_allocs = stacked_input_interleave * tile_cfg.K_outer_tile_count

    alignment_overhead = (
        num_filter_allocs * filter_align_waste
        + num_bias_allocs * bias_align_waste
        + num_result_allocs * result_align_waste
        + num_input_window_allocs * input_window_align_waste
        + num_stacked_input_allocs * stacked_input_align_waste
    )

    total_memory_required = (
        w_out_interleave * c_out_interleave * per_c_out_tile_inner
        + input_window_interleave * input_window_memory
        + stacked_input_interleave * stacked_input_memory
        + c_out_interleave * per_c_out_tile_outer
        + alignment_overhead
        + bn_memory
    )

    # Store-batch merging (column tiling, B > 1): merge each band's store across store_batch_merge
    # batches via one strided DMA; needs full flushes (D_out == 1, dh_range % num_dh_stacked == 0).
    store_batch_merge = 1
    store_pipe = w_out_interleave
    dh_range = tile_cfg.dh_end - tile_cfg.dh_start
    merge_ok = (
        tile_cfg.C_OUT_REP > 1
        and cfg.B > 1
        and w_out_interleave >= 2
        and cfg.D_out == 1
        and tile_cfg.num_dh_stacked > 0
        and dh_range % tile_cfg.num_dh_stacked == 0
    )
    if merge_ok:
        store_pipe = 2
        store_batch_merge = min(cfg.B, w_out_interleave // store_pipe)
        if store_batch_merge < 1:
            store_batch_merge, store_pipe = 1, w_out_interleave

    # Upper bound on rescale-phase reload interleave (# reload iterations that exist); over-approx
    # B x C_out tiles x spatial tiles, divided by rep_step since each iteration fills C_OUT_REP bands.
    bn_reload_interleave = 1
    if cfg.fuse_batch_norm:
        reload_free_tile = tile_cfg.num_dh_stacked * tile_cfg.W_tile
        total_spatial = cfg.D_out * cfg.H_out * cfg.W_out
        rep_step = tile_cfg.C_OUT_REP * reload_free_tile
        max_reload_iters = cfg.B * full_c_out_tile_count * div_ceil(total_spatial, rep_step)
        bn_reload_interleave = max(1, max_reload_iters)

    mem_cfg = Conv3dMemoryConfig(
        dtype_size=dtype_size,
        TOTAL_SBUF=TOTAL_SBUF,
        bias_size_per_tile=bias_size_per_tile,
        filters_per_c_out_tile=filters_per_c_out_tile,
        per_c_out_tile_outer=per_c_out_tile_outer,
        per_c_out_tile_inner=per_c_out_tile_inner,
        input_window_memory=input_window_memory,
        stacked_input_memory=stacked_input_memory,
        c_out_interleave=c_out_interleave,
        input_window_interleave=input_window_interleave,
        stacked_input_interleave=stacked_input_interleave,
        w_out_interleave=w_out_interleave,
        num_w_out_tiles=num_w_out_tiles,
        total_memory_required=total_memory_required,
        store_batch_merge=store_batch_merge,
        store_pipe=store_pipe,
        bn_reload_interleave=bn_reload_interleave,
    )

    logger = get_logger("conv3d")
    logger.info(
        f"MemoryConfig: c_out_interleave={mem_cfg.c_out_interleave}, "
        f"w_out_interleave={mem_cfg.w_out_interleave}, "
        f"input_window_interleave={mem_cfg.input_window_interleave}, "
        f"stacked_input_interleave={mem_cfg.stacked_input_interleave}, "
        f"store_batch_merge={mem_cfg.store_batch_merge}, store_pipe={mem_cfg.store_pipe}, "
        f"alignment_overhead={alignment_overhead}, "
        f"total_memory_required={mem_cfg.total_memory_required}, TOTAL_SBUF={mem_cfg.TOTAL_SBUF}"
    )

    return mem_cfg


def _get_tensor_copy_engine(idx: int, modulo: int = 4) -> Any:
    """Alternate between vector and scalar engines for tensor copy pipelining.

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    return nisa.scalar_engine if idx % modulo == 0 else nisa.vector_engine


def _get_memset_engine(idx: int, modulo: int = 2) -> Any:
    """Alternate between gpsimd and vector engines for memset pipelining.

    TODO: Determine optimal modulo using workload characteristics and cost model.
    """
    return nisa.gpsimd_engine if idx % modulo == 0 else nisa.vector_engine


def _validate_conv3d_inputs(
    x_in: nl.NkiTensor,
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    gamma: Optional[nl.NkiTensor] = None,
    beta: Optional[nl.NkiTensor] = None,
    running_means: Optional[nl.NkiTensor] = None,
    running_variances: Optional[nl.NkiTensor] = None,
) -> None:
    """Validate all input parameters for the 3D convolution kernel."""
    if cfg.fuse_batch_norm:
        kernel_assert(
            gamma != None and beta != None,
            "fuse_batch_norm=True requires both gamma and beta to be provided.",
        )
        kernel_assert(
            tuple(gamma.shape) == (cfg.C_out, 1),
            f"gamma must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(gamma.shape)}",
        )
        kernel_assert(
            tuple(beta.shape) == (cfg.C_out, 1),
            f"beta must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(beta.shape)}",
        )
        kernel_assert(
            running_means != None and running_variances != None,
            "fuse_batch_norm=True requires both running_means and running_variances to be provided.",
        )
        kernel_assert(
            tuple(running_means.shape) == (cfg.C_out, 1),
            f"running_means must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(running_means.shape)}",
        )
        kernel_assert(
            tuple(running_variances.shape) == (cfg.C_out, 1),
            f"running_variances must have shape [C_out, 1] = [{cfg.C_out}, 1], got {tuple(running_variances.shape)}",
        )
    kernel_assert(cfg.dilation_d >= 1, f"Dilation D must be >= 1, got {cfg.dilation_d}")
    kernel_assert(cfg.dilation_h >= 1, f"Dilation H must be >= 1, got {cfg.dilation_h}")
    kernel_assert(cfg.dilation_w >= 1, f"Dilation W must be >= 1, got {cfg.dilation_w}")
    kernel_assert(cfg.stride_d >= 1, f"Stride D must be >= 1, got {cfg.stride_d}")
    kernel_assert(cfg.stride_h >= 1, f"Stride H must be >= 1, got {cfg.stride_h}")
    kernel_assert(cfg.stride_w >= 1, f"Stride W must be >= 1, got {cfg.stride_w}")
    kernel_assert(
        cfg.pad_d_left >= 0 and cfg.pad_d_right >= 0,
        f"Padding D non-negative: ({cfg.pad_d_left}, {cfg.pad_d_right})",
    )
    kernel_assert(
        cfg.pad_h_top >= 0 and cfg.pad_h_bottom >= 0,
        f"Padding H non-negative: ({cfg.pad_h_top}, {cfg.pad_h_bottom})",
    )
    kernel_assert(
        cfg.pad_w_left >= 0 and cfg.pad_w_right >= 0,
        f"Padding W non-negative: ({cfg.pad_w_left}, {cfg.pad_w_right})",
    )
    kernel_assert(cfg.D_out >= 1, f"Output depth D_out must be >= 1, got {cfg.D_out}")
    kernel_assert(cfg.H_out >= 1, f"Output height H_out must be >= 1, got {cfg.H_out}")
    kernel_assert(cfg.W_out >= 1, f"Output width W_out must be >= 1, got {cfg.W_out}")
    kernel_assert(x_in.shape[1] == filters.shape[3], f"C_in mismatch: {x_in.shape[1]} vs {filters.shape[3]}")
    kernel_assert(x_in.dtype == filters.dtype, f"dtype mismatch: {x_in.dtype} vs {filters.dtype}")
    if bias != None:
        kernel_assert(x_in.dtype == bias.dtype, f"bias dtype mismatch: {x_in.dtype} vs {bias.dtype}")


# Memory Hierarchy 3: SBUF operations


def _compute_valid_free_range(
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    used_k_positions: list[int],
    k_start: int,
    k_end: int,
) -> tuple[int, int]:
    """
    Compute the contiguous valid free-dim range for a K-outer tile.

    For each K position in [k_start, k_end), determines which (dh_idx, w_out)
    positions in the free dimension have at least one input position within valid
    bounds.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    full_free_dim = num_dh_stacked * w_tile_size

    # Start with empty range and take union
    global_valid_start = full_free_dim
    global_valid_end = 0

    for k_idx in range(k_start, k_end):
        flat_k_pos = used_k_positions[k_idx]
        k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)

        # Compute valid W range for this K position
        base_w_in_first = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        # First valid w_out offset: input position >= 0
        if base_w_in_first < 0:
            k_w_valid_start = div_ceil(-base_w_in_first, cfg.stride_w)
        else:
            k_w_valid_start = 0
        # Last valid w_out offset: input position < W
        base_w_in_last = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
        if base_w_in_last >= cfg.W:
            k_w_valid_end = w_tile_size - div_ceil(base_w_in_last - cfg.W + 1, cfg.stride_w)
        else:
            k_w_valid_end = w_tile_size

        if k_w_valid_start >= k_w_valid_end:
            # This K position has no valid W positions
            continue

        # Find first and last valid dh positions for this K position
        first_valid_dh = -1
        last_valid_dh = -1
        for dh_idx in range(num_dh_stacked):
            d_out, h_out = dh_positions[dh_idx]
            d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
            h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
            if 0 <= d_in < cfg.D and 0 <= h_in < cfg.H:
                if first_valid_dh == -1:
                    first_valid_dh = dh_idx
                last_valid_dh = dh_idx

        if first_valid_dh == -1:
            # No valid dh positions for this K position
            continue

        # Valid free-dim range: [first_valid_dh * w_tile_size + k_w_valid_start,
        #                        last_valid_dh * w_tile_size + k_w_valid_end)
        k_valid_start = first_valid_dh * w_tile_size + k_w_valid_start
        k_valid_end = last_valid_dh * w_tile_size + k_w_valid_end

        # Union with global range
        global_valid_start = min(global_valid_start, k_valid_start)
        global_valid_end = max(global_valid_end, k_valid_end)

    if global_valid_start >= global_valid_end:
        return 0, 0

    return global_valid_start, global_valid_end


def _compute_unwritten_free_ranges(
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> list[tuple[int, int]]:
    """
    Compute free-dim ranges that no nc_matmul will ever write for this output tile.
    """
    K_total_used = tile_cfg.K_total_used
    K_REP = tile_cfg.K_REP_max
    K_outer_tile_count = tile_cfg.K_outer_tile_count
    w_tile_size = w_end - w_start
    effective_free_dim = len(dh_positions) * w_tile_size

    # Written mask = union of each K-outer tile's valid range; False positions are never written.
    written = [False] * effective_free_dim
    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        valid_start, valid_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            tile_cfg.used_k_positions,
            k_start,
            k_end,
        )
        for pos in range(valid_start, valid_end):
            written[pos] = True

    # Coalesce contiguous unwritten positions into ranges
    unwritten_ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < effective_free_dim:
        if written[pos]:
            pos += 1
            continue
        gap_start = pos
        while pos < effective_free_dim and not written[pos]:
            pos += 1
        unwritten_ranges.append((gap_start, pos))
    return unwritten_ranges


def _scatter_input_to_stacked(
    input_window: Optional[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    memset_engine_idx: int = 0,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
) -> tuple[list[Optional[nl.NkiTensor]], list[int], list[int]]:
    """
    Scatter input window data into K-replicated stacked buffers for matmul.

    For each K-outer tile, maps the relevant input positions from the loaded
    input window into a stacked buffer layout where multiple filter positions
    are packed along the partition dimension. Handles padding by zero-initializing
    buffers or adjusting the free-dimension and stride of input tiles. Returns None
    for fully-padded K-outer tiles to skip unnecessary matmul operations.

    This function is agnostic to C_out column tiling: it always produces the stacked
    input covering the whole num_dh_stacked free dim. Column tiling is applied purely
    downstream in _conv3d_matmul, which slices this stacked input per column-tiling group.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    effective_free_dim = num_dh_stacked * w_tile_size
    K_REP, partition_stride = _get_k_replication_params(c_in_tile_size, K_total_used) if c_in_tile_size > 0 else (1, 1)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)
    input_stacked_list: list[Optional[nl.NkiTensor]] = []
    valid_offsets: list[int] = []
    valid_sizes: list[int] = []
    valid_dh_starts: list[int] = []
    valid_dh_counts: list[int] = []
    valid_w_starts: list[int] = []
    valid_w_counts: list[int] = []
    use_strided_flags: list[bool] = []

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = stacked_input_bufs[k_outer_idx]

        # Compute the valid free-dim range for this K-outer tile
        valid_free_start, valid_free_end = _compute_valid_free_range(
            cfg,
            dh_positions,
            w_start,
            w_end,
            used_k_positions,
            k_start,
            k_end,
        )

        if valid_free_start >= valid_free_end:
            input_stacked_list.append(None)
            valid_offsets.append(0)
            valid_sizes.append(0)
            valid_dh_starts.append(0)
            valid_dh_counts.append(0)
            valid_w_starts.append(0)
            valid_w_counts.append(0)
            use_strided_flags.append(False)
            continue

        valid_free_size = valid_free_end - valid_free_start
        first_valid_dh_idx_pre = valid_free_start // w_tile_size
        last_valid_dh_idx_pre = (valid_free_end - 1) // w_tile_size
        k_w_valid_start_pre = valid_free_start - first_valid_dh_idx_pre * w_tile_size
        k_w_valid_end_pre = valid_free_end - last_valid_dh_idx_pre * w_tile_size
        valid_dh_count_pre = last_valid_dh_idx_pre - first_valid_dh_idx_pre + 1
        has_w_padding = k_w_valid_start_pre > 0 or k_w_valid_end_pre < w_tile_size

        # All K positions share the same W valid range?
        all_k_same_w_range_pre = True
        if k_actual > 1 and valid_dh_count_pre > 1 and has_w_padding:
            for k_check_pre in range(k_actual):
                flat_k_pre = used_k_positions[k_start + k_check_pre]
                _, _, k_w_pre = _decompose_k_position(flat_k_pre, K_h, K_w)
                base_w_pre = w_start * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_start_pre = div_ceil(-base_w_pre, cfg.stride_w) if base_w_pre < 0 else 0
                last_w_pre = (w_end - 1) * cfg.stride_w + k_w_pre * cfg.dilation_w - cfg.pad_w_left
                this_w_end_pre = (
                    w_tile_size - div_ceil(last_w_pre - cfg.W + 1, cfg.stride_w) if last_w_pre >= cfg.W else w_tile_size
                )
                if this_w_start_pre != k_w_valid_start_pre or this_w_end_pre != k_w_valid_end_pre:
                    all_k_same_w_range_pre = False
                    break

        will_use_strided = all_k_same_w_range_pre and valid_dh_count_pre > 1 and has_w_padding

        # Check if any positions within the valid free-dim range need zero memset.
        needs_zero_init = partition_stride > c_in_tile_size
        if not needs_zero_init:
            first_valid_dh_idx = valid_free_start // w_tile_size
            last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
            for dh_idx in range(first_valid_dh_idx, last_valid_dh_idx + 1):
                d_out, h_out = dh_positions[dh_idx]
                # Portion of this dh's W range within the valid range.
                dh_free_start = dh_idx * w_tile_size
                dh_free_end = (dh_idx + 1) * w_tile_size
                # Clip to the valid range
                check_w_start = max(dh_free_start, valid_free_start) - dh_free_start
                check_w_end = min(dh_free_end, valid_free_end) - dh_free_start
                for k_rep_idx in range(k_actual):
                    flat_k_pos = used_k_positions[k_start + k_rep_idx]
                    k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < 0 or d_in >= cfg.D or h_in < 0 or h_in >= cfg.H:
                        needs_zero_init = True
                        break
                    # W range within the valid portion has any padding?
                    if not will_use_strided:
                        base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        first_valid_w = 0 if base_w_in >= 0 else div_ceil(-base_w_in, cfg.stride_w)
                        last_w_in = (w_end - 1) * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left
                        last_valid_w = (
                            w_tile_size
                            if last_w_in < cfg.W
                            else w_tile_size - div_ceil(last_w_in - cfg.W + 1, cfg.stride_w)
                        )
                        if first_valid_w > check_w_start or last_valid_w < check_w_end:
                            needs_zero_init = True
                            break
                if needs_zero_init:
                    break

        if needs_zero_init:
            # Memset only the valid-range portion.
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        if input_window != None:
            for k_rep_idx in range(k_actual):
                partition_offset = k_rep_idx * partition_stride
                flat_k_pos = used_k_positions[k_start + k_rep_idx]
                k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
                base_w_in = w_start * cfg.stride_w + k_w_idx * cfg.dilation_w - cfg.pad_w_left

                # Compute valid W range
                first_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_start - base_w_in, cfg.stride_w) if base_w_in < valid_w_start else 0,
                        w_tile_size,
                    ),
                )
                last_valid_out = max(
                    0,
                    min(
                        div_ceil(valid_w_end - base_w_in, cfg.stride_w)
                        if base_w_in + (w_tile_size - 1) * cfg.stride_w >= valid_w_end
                        else w_tile_size,
                        w_tile_size,
                    ),
                )

                if first_valid_out >= last_valid_out:
                    continue

                src_w_start = base_w_in + first_valid_out * cfg.stride_w - valid_w_start
                num_w_elements = last_valid_out - first_valid_out
                src_w_end = src_w_start + (num_w_elements - 1) * cfg.stride_w + 1

                # Group consecutive dh_idx values into fewer, larger strided tensor_copy ops.
                dh_idx = 0
                while dh_idx < num_dh_stacked:
                    d_out, h_out = dh_positions[dh_idx]
                    d_in = d_out * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                    h_in = h_out * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                    if d_in < valid_d_start or d_in >= valid_d_end or h_in < valid_h_start or h_in >= valid_h_end:
                        dh_idx += 1
                        continue

                    d_local_start = d_in - valid_d_start
                    h_local_start = h_in - valid_h_start
                    group_start_dh = dh_idx

                    # Group consecutive H positions with the same d_in
                    num_h = 1
                    next_dh = dh_idx + 1
                    while next_dh < num_dh_stacked:
                        d_out_next, h_out_next = dh_positions[next_dh]
                        d_in_next = d_out_next * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_next = h_out_next * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        if d_in_next != d_in:
                            break
                        if h_in_next < valid_h_start or h_in_next >= valid_h_end:
                            break
                        expected_h_in = h_in + num_h * cfg.stride_h
                        if h_in_next != expected_h_in:
                            break
                        num_h += 1
                        next_dh += 1

                    # Extend across D: each D group needs num_h consecutive H at the same
                    # h_local_start, d_in advancing by stride_d.
                    num_d = 1
                    while next_dh + num_h - 1 < num_dh_stacked:
                        # Next num_h positions form a valid D+H group?
                        d_out_check, h_out_check = dh_positions[next_dh]
                        d_in_check = d_out_check * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                        h_in_check = h_out_check * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                        expected_d_in = d_in + num_d * cfg.stride_d
                        if d_in_check != expected_d_in:
                            break
                        if d_in_check < valid_d_start or d_in_check >= valid_d_end:
                            break
                        if h_in_check != h_in:
                            break
                        # All num_h positions in this D group match?
                        valid_d_group = True
                        for h_check_idx in range(1, num_h):
                            d_out_c, h_out_c = dh_positions[next_dh + h_check_idx]
                            d_in_c = d_out_c * cfg.stride_d + k_d_idx * cfg.dilation_d - cfg.pad_d_left
                            h_in_c = h_out_c * cfg.stride_h + k_h_idx * cfg.dilation_h - cfg.pad_h_top
                            if d_in_c != expected_d_in:
                                valid_d_group = False
                                break
                            expected_h_c = h_in + h_check_idx * cfg.stride_h
                            if h_in_c != expected_h_c:
                                valid_d_group = False
                                break
                            if h_in_c < valid_h_start or h_in_c >= valid_h_end:
                                valid_d_group = False
                                break
                        if not valid_d_group:
                            break
                        num_d += 1
                        next_dh += num_h

                    total_dh_in_group = num_d * num_h

                    if total_dh_in_group == 1:
                        # Single dh position, 1D strided copy
                        dst_start = group_start_dh * w_tile_size + first_valid_out
                        dst_end = dst_start + num_w_elements
                        nisa.tensor_copy(
                            dst=raw_buf[partition_offset : partition_offset + c_in_tile_size, dst_start:dst_end],
                            src=input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .select(dim=1, index=h_local_start)
                            .slice(dim=1, start=src_w_start, end=src_w_end, step=cfg.stride_w),
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    elif num_d == 1:
                        # Multiple H positions, 2D strided copy
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .select(dim=1, index=d_local_start)
                            .slice(dim=1, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=2, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + num_h,
                                step=1,
                            )
                            .slice(dim=2, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )
                    else:
                        # Multiple D and H positions, 3D strided copy
                        d_local_end = d_local_start + (num_d - 1) * cfg.stride_d + 1
                        h_local_end = h_local_start + (num_h - 1) * cfg.stride_h + 1
                        src_view = (
                            input_window.slice(dim=0, start=0, end=c_in_tile_size, step=1)
                            .slice(dim=1, start=d_local_start, end=d_local_end, step=cfg.stride_d)
                            .slice(dim=2, start=h_local_start, end=h_local_end, step=cfg.stride_h)
                            .slice(dim=3, start=src_w_start, end=src_w_end, step=cfg.stride_w)
                        )
                        dst_view = (
                            raw_buf.slice(
                                dim=0,
                                start=partition_offset,
                                end=partition_offset + c_in_tile_size,
                                step=1,
                            )
                            .slice(dim=1, start=0, end=effective_free_dim, step=1)
                            .reshape_dim(1, (num_dh_stacked, w_tile_size))
                            .slice(
                                dim=1,
                                start=group_start_dh,
                                end=group_start_dh + total_dh_in_group,
                                step=1,
                            )
                            .reshape_dim(1, (num_d, num_h))
                            .slice(dim=3, start=first_valid_out, end=last_valid_out, step=1)
                        )
                        nisa.tensor_copy(
                            dst=dst_view,
                            src=src_view,
                            engine=_get_tensor_copy_engine(
                                tensor_copy_engine_idx
                                + group_start_dh * K_total_used
                                + k_outer_idx * K_REP
                                + k_rep_idx,
                                tensor_copy_engine_modulo,
                            ),
                        )

                    dh_idx = next_dh

        # Per-K dh and W valid ranges for strided matmul (as in _compute_valid_free_range).
        first_valid_dh_idx, k_w_valid_start_val = divmod(valid_free_start, w_tile_size)
        last_valid_dh_idx = (valid_free_end - 1) // w_tile_size
        k_w_valid_end_val = valid_free_end - last_valid_dh_idx * w_tile_size
        valid_dh_count = last_valid_dh_idx - first_valid_dh_idx + 1
        valid_w_count = k_w_valid_end_val - k_w_valid_start_val

        # Strided matmul requires all K positions in the tile to share a W valid range
        # (trivially true for K_REP=1).
        all_k_same_w_range = True
        if k_actual > 1 and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size):
            ref_w_start_check = k_w_valid_start_val
            ref_w_end_check = k_w_valid_end_val
            for k_check_idx in range(k_actual):
                flat_k_check = used_k_positions[k_start + k_check_idx]
                _, _, k_w_check = _decompose_k_position(flat_k_check, K_h, K_w)
                base_w_check = w_start * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if base_w_check < 0:
                    this_w_start = div_ceil(-base_w_check, cfg.stride_w)
                else:
                    this_w_start = 0
                last_w_check = (w_end - 1) * cfg.stride_w + k_w_check * cfg.dilation_w - cfg.pad_w_left
                if last_w_check >= cfg.W:
                    this_w_end = w_tile_size - div_ceil(last_w_check - cfg.W + 1, cfg.stride_w)
                else:
                    this_w_end = w_tile_size
                if this_w_start != ref_w_start_check or this_w_end != ref_w_end_check:
                    all_k_same_w_range = False
                    break

        use_strided = (
            all_k_same_w_range and valid_dh_count > 1 and (k_w_valid_start_val > 0 or k_w_valid_end_val < w_tile_size)
        )

        # Strided view of the stacked buffer, matching the PSUM strided view.
        if use_strided:
            # Strided case
            input_strided = (
                raw_buf.slice(dim=0, start=0, end=stacked_filter_dim, step=1)
                .slice(dim=1, start=0, end=effective_free_dim, step=1)
                .reshape_dim(1, (num_dh_stacked, w_tile_size))
                .slice(dim=1, start=first_valid_dh_idx, end=first_valid_dh_idx + valid_dh_count, step=1)
                .slice(dim=2, start=k_w_valid_start_val, end=k_w_valid_end_val, step=1)
            )
            input_stacked_list.append(input_strided)
        else:
            # Contiguous case
            input_stacked_list.append(raw_buf[:stacked_filter_dim, valid_free_start:valid_free_end])

        valid_offsets.append(valid_free_start)
        valid_sizes.append(valid_free_size)
        valid_dh_starts.append(first_valid_dh_idx)
        valid_dh_counts.append(valid_dh_count)
        valid_w_starts.append(k_w_valid_start_val)
        valid_w_counts.append(valid_w_count)
        use_strided_flags.append(use_strided)
    return (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        valid_dh_starts,
        valid_dh_counts,
        valid_w_starts,
        valid_w_counts,
        use_strided_flags,
    )


def _conv3d_matmul(
    input_stacked_list: list[Optional[nl.NkiTensor]],
    filters_stacked_list: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    valid_offsets: list[int],
    valid_sizes: list[int],
    effective_free_dim: int,
    num_dh_stacked: int,
    w_tile_size: int,
    valid_dh_starts: list[int],
    valid_dh_counts: list[int],
    valid_w_starts: list[int],
    valid_w_counts: list[int],
    use_strided_flags: list[bool],
    c_out_tile_sizes: Optional[list[int]] = None,
    C_OUT_REP: int = 1,
    col_size: int = 128,
    dh_head_size: int = 0,
) -> None:
    """
    Accumulate matmul results across K-outer tiles into PSUM for all C_out tiles.

    Uses strided 2D views to handle W-padding without memsets: reshapes the free
    dimension as (num_dh, w_tile) and slices both the dh and W dimensions to
    exclude padded positions. This creates a strided access pattern where each
    dh row contributes only valid_w_count elements with stride w_tile_size,
    matching the flat compiler's approach.

    Tensor Engine column tiling (C_OUT_REP > 1): the num_dh_stacked = C_OUT_REP * dh_head_size
    D-H positions of a flush form C_OUT_REP bands of dh_head_size positions each. Every K-outer
    matmul is distributed across the bands: band g's matmul processes only its D-H window,
    keeps the full dh_head_size * W_tile free dim, runs on PE column tile (0, g * col_size), and
    writes to PSUM partition band [g * col_size, g * col_size + M). The stacked input from
    _scatter_input_to_stacked is reused unchanged (column tiling only slices it downstream), so
    the same full / contiguous / strided-W-padding access patterns apply per band. For
    C_OUT_REP == 1 there is a single band spanning the whole flush at partition 0 with no PE
    column tiling, i.e. byte-identical to the non-column-tiled behavior.
    """
    bands = _column_bands(num_dh_stacked, dh_head_size if C_OUT_REP > 1 else num_dh_stacked, C_OUT_REP, col_size)

    for k_outer_idx in range(len(input_stacked_list)):
        input_stacked = input_stacked_list[k_outer_idx]
        if input_stacked == None:
            continue
        k_offset = valid_offsets[k_outer_idx]
        k_size = valid_sizes[k_outer_idx]
        dh_start = valid_dh_starts[k_outer_idx]
        dh_count = valid_dh_counts[k_outer_idx]
        w_start_k = valid_w_starts[k_outer_idx]
        w_count = valid_w_counts[k_outer_idx]
        use_strided = use_strided_flags[k_outer_idx]
        for c_out_idx in range(len(psum_tiles)):
            stationary = filters_stacked_list[c_out_idx][k_outer_idx]
            for band_idx in range(len(bands)):
                part_off = bands[band_idx][0]
                band_dh_lo = bands[band_idx][1]
                band_dh_hi = bands[band_idx][2]
                col_pos = part_off if C_OUT_REP > 1 else None
                _conv3d_matmul_band(
                    input_stacked=input_stacked,
                    stationary=stationary,
                    psum_tile=psum_tiles[c_out_idx],
                    k_offset=k_offset,
                    k_size=k_size,
                    dh_start=dh_start,
                    dh_count=dh_count,
                    w_start_k=w_start_k,
                    w_count=w_count,
                    use_strided=use_strided,
                    effective_free_dim=effective_free_dim,
                    w_tile_size=w_tile_size,
                    part_off=part_off,
                    band_dh_lo=band_dh_lo,
                    band_dh_hi=band_dh_hi,
                    col_pos=col_pos,
                    col_size=col_size,
                )


def _flush_column_layout(tile_cfg: "Conv3dTileConfig", num_dh_positions: int) -> tuple[int, int, int]:
    """
    Decide the column-tiling layout for one flush of num_dh_positions D-H positions. Returns
    (col_rep, band_dh, col_size): col_rep = active bands (1 disables tiling), band_dh = per-band
    D-H count, col_size = PE column-tile / partition-band stride (P_MAX when col_rep == 1).

    Tiling applies when the flush spans > dh_head_size positions; band_dh stays dh_head_size (the
    last band may be partial) so every eviction fits the per-band result width. A rep-1 fallback
    would size the eviction at num_dh_positions * W_tile and overflow the band-width buffer. Shared
    by _conv3d_output_tile and the store loop so they agree on the layout.
    """
    dh_head = tile_cfg.dh_head_size
    if tile_cfg.C_OUT_REP > 1 and dh_head > 0 and num_dh_positions > dh_head:
        col_rep = min(tile_cfg.C_OUT_REP, div_ceil(num_dh_positions, dh_head))
        return col_rep, dh_head, tile_cfg.col_size
    return 1, num_dh_positions, tile_cfg.P_MAX


def _column_bands(
    num_dh_positions: int, dh_head_size: int, C_OUT_REP: int, col_size: int
) -> list[tuple[int, int, int]]:
    """
    Enumerate the column-tiling bands as (partition_offset, dh_lo, dh_hi). Band band_idx covers dh
    [band_idx * dh_head_size, min((band_idx+1) * dh_head_size, num_dh_positions)) at partition offset
    band_idx * col_size. A tail flush may activate fewer than C_OUT_REP bands (last one partial);
    C_OUT_REP == 1 is a single band at partition 0 over all positions.
    """
    if C_OUT_REP <= 1:
        return [(0, 0, num_dh_positions)]
    bands: list[tuple[int, int, int]] = []
    for band_idx in range(C_OUT_REP):
        dh_lo = band_idx * dh_head_size
        if dh_lo >= num_dh_positions:
            break
        dh_hi = min(dh_lo + dh_head_size, num_dh_positions)
        bands.append((band_idx * col_size, dh_lo, dh_hi))
    return bands


def _conv3d_matmul_band(
    input_stacked: nl.NkiTensor,
    stationary: nl.NkiTensor,
    psum_tile: nl.NkiTensor,
    k_offset: int,
    k_size: int,
    dh_start: int,
    dh_count: int,
    w_start_k: int,
    w_count: int,
    use_strided: bool,
    effective_free_dim: int,
    w_tile_size: int,
    part_off: int,
    band_dh_lo: int,
    band_dh_hi: int,
    col_pos: Optional[int],
    col_size: int,
) -> None:
    """
    Issue one K-outer matmul for a single column-tiling band.

    Restricts the K-outer tile's valid region (from the scatter, expressed over the whole flush)
    to this band's D-H window [band_dh_lo, band_dh_hi), remaps it to the band's local PSUM free
    coordinates (band-local free position = global free position - band_dh_lo * w_tile_size), and
    writes to PSUM partitions [part_off, part_off + M) where M is the stationary free (C_out)
    dim. When col_pos is not None the matmul selects PE column tile (0, col_pos) with tile_size
    (128, col_size) so bands run on disjoint PE columns.

    Reuses the same three access patterns as the non-column-tiled matmul (full free / contiguous
    sub-range / strided W-padding). For a single band spanning the whole flush (C_OUT_REP == 1,
    part_off == 0, col_pos is None), every branch reduces to the original non-tiled behavior.
    """
    band_free_lo = band_dh_lo * w_tile_size
    band_free_hi = band_dh_hi * w_tile_size
    M = stationary.shape[1]
    use_tiling = col_pos != None

    if not use_strided:
        # Flat access: intersect the contiguous valid range [k_offset, k_offset + k_size) with the
        # band's flat window.
        a = max(k_offset, band_free_lo)
        b = min(k_offset + k_size, band_free_hi)
        if a >= b:
            return
        moving = input_stacked.slice(dim=1, start=a - k_offset, end=b - k_offset, step=1)
        psum_sub = (
            (psum_tile)
            .slice(dim=0, start=part_off, end=part_off + M, step=1)
            .slice(dim=1, start=a - band_free_lo, end=b - band_free_lo, step=1)
        )
    else:
        # Strided access: input_stacked is a 3D (contraction, dh, w) view; intersect its dh range
        # [dh_start, dh_start + dh_count) with the band's dh window.
        dh_a = max(dh_start, band_dh_lo)
        dh_b = min(dh_start + dh_count, band_dh_hi)
        if dh_a >= dh_b:
            return
        moving = input_stacked.slice(dim=1, start=dh_a - dh_start, end=dh_b - dh_start, step=1)
        band_dh = band_dh_hi - band_dh_lo
        psum_sub = (
            (psum_tile)
            .slice(dim=0, start=part_off, end=part_off + M, step=1)
            .slice(dim=1, start=0, end=band_dh * w_tile_size, step=1)
            .reshape_dim(1, (band_dh, w_tile_size))
            .slice(dim=1, start=dh_a - band_dh_lo, end=dh_b - band_dh_lo, step=1)
            .slice(dim=2, start=w_start_k, end=w_start_k + w_count, step=1)
        )

    if use_tiling:
        nisa.nc_matmul(
            dst=psum_sub,
            stationary=stationary,
            moving=moving,
            tile_position=(0, col_pos),
            tile_size=(nl.tile_size.pmax, col_size),
        )
    else:
        nisa.nc_matmul(dst=psum_sub, stationary=stationary, moving=moving)


def _conv3d_cin_tile(
    input_window: Optional[nl.NkiTensor],
    filters_for_cin: list[list[nl.NkiTensor]],
    psum_tiles: list[nl.NkiTensor],
    stacked_input_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_tile_size: int,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    valid_d_start: int,
    valid_d_end: int,
    valid_h_start: int,
    valid_h_end: int,
    valid_w_start: int,
    valid_w_end: int,
    used_k_positions: list[int],
    tensor_copy_engine_idx: int,
    effective_free_dim: int,
    tensor_copy_engine_modulo: int = 4,
    memset_engine_modulo: int = 2,
    c_out_tile_sizes: Optional[list[int]] = None,
    C_OUT_REP: int = 1,
    col_size: int = 128,
    dh_head_size: int = 0,
) -> None:
    """
    Process one C_in tile: scatter input into stacked layout and accumulate matmul into PSUM.

    Combines the scatter and matmul steps for a single input channel tile. First
    scatters the input window into K-replicated stacked buffers, then performs
    matmul accumulation across all C_out tiles. When C_OUT_REP > 1 the matmul is
    column-tiled across C_OUT_REP PE column tiles (see _conv3d_matmul).
    """
    w_tile_size = w_end - w_start
    num_dh_stacked = len(dh_positions)
    (
        input_stacked_list,
        valid_offsets,
        valid_sizes,
        v_dh_starts,
        v_dh_counts,
        v_w_starts,
        v_w_counts,
        v_strided_flags,
    ) = _scatter_input_to_stacked(
        input_window=input_window,
        stacked_input_bufs=stacked_input_bufs,
        cfg=cfg,
        c_in_tile_size=c_in_tile_size,
        dh_positions=dh_positions,
        w_start=w_start,
        w_end=w_end,
        valid_d_start=valid_d_start,
        valid_d_end=valid_d_end,
        valid_h_start=valid_h_start,
        valid_h_end=valid_h_end,
        valid_w_start=valid_w_start,
        valid_w_end=valid_w_end,
        used_k_positions=used_k_positions,
        tensor_copy_engine_idx=tensor_copy_engine_idx,
        tensor_copy_engine_modulo=tensor_copy_engine_modulo,
        memset_engine_modulo=memset_engine_modulo,
    )
    _conv3d_matmul(
        input_stacked_list=input_stacked_list,
        filters_stacked_list=filters_for_cin,
        psum_tiles=psum_tiles,
        valid_offsets=valid_offsets,
        valid_sizes=valid_sizes,
        effective_free_dim=effective_free_dim,
        num_dh_stacked=num_dh_stacked,
        w_tile_size=w_tile_size,
        valid_dh_starts=v_dh_starts,
        valid_dh_counts=v_dh_counts,
        valid_w_starts=v_w_starts,
        valid_w_counts=v_w_counts,
        use_strided_flags=v_strided_flags,
        c_out_tile_sizes=c_out_tile_sizes,
        C_OUT_REP=C_OUT_REP,
        col_size=col_size,
        dh_head_size=dh_head_size,
    )


# Memory Hierarchy 2: HBM <-> SBUF interleaved with SBUF operations


def _band_free_gap(
    band_dh_lo: int, w_tile_size: int, effective_free_dim: int, gap_start: int, gap_end: int
) -> tuple[int, int]:
    """
    Translate a flush-global never-written free range into one band's band-local free coords.

    Band positions start at flush-global free position band_dh_lo * w_tile_size and span
    effective_free_dim columns. Returns (lo, hi) clipped to the band (lo >= hi means the gap
    does not touch this band).
    """
    band_free_lo = band_dh_lo * w_tile_size
    lo = max(gap_start, band_free_lo) - band_free_lo
    hi = min(gap_end, band_free_lo + effective_free_dim) - band_free_lo
    return lo, hi


def _memset_band_free_gaps(
    psum_tile: nl.NkiTensor,
    column_bands: list[tuple[int, int, int]],
    c_out_tile_size: int,
    unwritten_free_ranges: list[tuple[int, int]],
    effective_free_dim: int,
    w_tile_size: int,
    memset_engine_modulo: int,
) -> None:
    """Zero the never-written free positions of each band directly in PSUM (batchnorm path)."""
    gap_engine_ctr = 0
    for band_idx in range(len(column_bands)):
        bp_lo = column_bands[band_idx][0]
        bp_hi = bp_lo + c_out_tile_size
        band_dh_lo = column_bands[band_idx][1]
        for gap_idx in range(len(unwritten_free_ranges)):
            lo, hi = _band_free_gap(
                band_dh_lo,
                w_tile_size,
                effective_free_dim,
                unwritten_free_ranges[gap_idx][0],
                unwritten_free_ranges[gap_idx][1],
            )
            if lo < hi:
                nisa.memset(
                    dst=psum_tile[bp_lo:bp_hi, lo:hi],
                    value=0.0,
                    engine=_get_memset_engine(gap_engine_ctr, memset_engine_modulo),
                )
                gap_engine_ctr += 1


def _fill_band_free_gaps_sbuf(
    result_full: nl.NkiTensor,
    bias_sbuf: Optional[nl.NkiTensor],
    column_bands: list[tuple[int, int, int]],
    c_out_tile_size: int,
    unwritten_free_ranges: list[tuple[int, int]],
    effective_free_dim: int,
    w_tile_size: int,
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
    memset_engine_modulo: int,
) -> None:
    """
    Fill never-written free positions in the result SBUF with f(0 + bias) (non-batchnorm path).

    Per band, zero the band-local gap then apply the same bias / activation as the main copy.
    The bias for band g lives at partitions [g * col_size, ...) of the passed bias_sbuf (it is
    pre-replicated across bands), matching the band's result partitions.
    """
    gap_engine_ctr = 0
    for band_idx in range(len(column_bands)):
        bp_lo = column_bands[band_idx][0]
        bp_hi = bp_lo + c_out_tile_size
        band_dh_lo = column_bands[band_idx][1]
        band_bias = bias_sbuf[bp_lo:bp_hi, :] if (has_bias and bias_sbuf != None) else bias_sbuf
        for gap_idx in range(len(unwritten_free_ranges)):
            lo, hi = _band_free_gap(
                band_dh_lo,
                w_tile_size,
                effective_free_dim,
                unwritten_free_ranges[gap_idx][0],
                unwritten_free_ranges[gap_idx][1],
            )
            if lo >= hi:
                continue
            gap_result = result_full[bp_lo:bp_hi, lo:hi]
            nisa.memset(dst=gap_result, value=0.0, engine=_get_memset_engine(gap_engine_ctr, memset_engine_modulo))
            gap_engine_ctr += 1
            if has_bias and has_activation:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=band_bias)
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))
            elif has_bias:
                nisa.tensor_scalar(dst=gap_result, data=gap_result, op0=nl.add, operand0=band_bias)
            elif has_activation:
                nisa.activation(dst=gap_result, data=gap_result, op=get_nl_act_fn_from_type(activation_fn))


def _apply_bias_activation_and_copy(
    psum_tiles: list[nl.NkiTensor],
    result_sbufs: list[nl.NkiTensor],
    bias_sbufs: list[Optional[nl.NkiTensor]],
    c_out_tile_sizes: list[int],
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
    effective_free_dim: int,
    unwritten_free_ranges: Optional[list[tuple[int, int]]] = None,
    memset_engine_modulo: int = 2,
    fuse_batch_norm: bool = False,
    bn_sum_acc: Optional[nl.NkiTensor] = None,
    bn_sq_scratch: Optional[nl.NkiTensor] = None,
    bn_sqsum_partials: Optional[nl.NkiTensor] = None,
    bn_partial_iters: int = 1,
    bn_iter_idx: int = 0,
    bn_tile_offset: int = 0,
    evict_partitions: int = 0,
    column_bands: Optional[list[tuple[int, int, int]]] = None,
    w_tile_size: int = 1,
) -> None:
    """
    Apply optional bias / activation and copy the PSUM result to SBUF. Two paths:

    * fuse_batch_norm=False: bias and/or activation on the copy-out (or a plain copy), filling
      never-written padded positions with f(0 + bias).
    * fuse_batch_norm=True: fuse the copy + bias + per-channel sum(Y) / sum(Y^2) reduction, with no
      activation (deferred to the rescale phase so stats see the raw output). sum(Y) accumulates in
      place into bn_sum_acc (load_reduce); sum(Y^2) goes to a per-flush column of bn_sqsum_partials
      (reduced later) to avoid a per-flush RMW. See conv3d_design_spec.md.

    Column tiling: the flush's C_OUT_REP bands share the same band-local free layout
    [0, effective_free_dim) at partition offsets g * col_size, so a single op over
    [0, evict_partitions) evicts all bands at once (the per-band split happens downstream).
    C_OUT_REP == 1 is a single band, byte-identical to the non-tiled path.
    """
    has_unwritten = unwritten_free_ranges != None and len(unwritten_free_ranges) > 0
    if column_bands == None:
        column_bands = [(0, 0, 0)]

    for c_out_tile_idx in range(len(result_sbufs)):
        psum_tile = psum_tiles[c_out_tile_idx]
        result_full = result_sbufs[c_out_tile_idx]
        bias_full = bias_sbufs[c_out_tile_idx]
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        # Active partition span over all bands (== c_out_tile_size when col_rep == 1). Bias is
        # pre-replicated across bands; slice it to evict_p (buffer may be taller on a rep-1 tail).
        evict_p = evict_partitions if evict_partitions > 0 else c_out_tile_size
        bias_sbuf = bias_full[:evict_p, :] if (has_bias and bias_full != None) else bias_full
        result = result_full[:evict_p, :effective_free_dim]
        psum_data = psum_tile[:evict_p, :effective_free_dim]

        if fuse_batch_norm:
            global_tile_idx = bn_tile_offset + c_out_tile_idx
            sum_col = bn_sum_acc[:evict_p, global_tile_idx : global_tile_idx + 1]
            # This flush's sum(Y^2) partial -> its own column (block of bn_partial_iters per C_out
            # tile); reduced later, so no per-flush RMW here.
            partial_col_idx = global_tile_idx * bn_partial_iters + bn_iter_idx
            partial_sqsum = bn_sqsum_partials[:evict_p, partial_col_idx : partial_col_idx + 1]

            # Zero never-written padded PSUM positions so the sums include their true f(0 + bias).
            if has_unwritten:
                _memset_band_free_gaps(
                    psum_tile,
                    column_bands,
                    c_out_tile_size,
                    unwritten_free_ranges,
                    effective_free_dim,
                    w_tile_size,
                    memset_engine_modulo,
                )

            # Ragged tail: zero a partial last band's unwritten free-column tail so the full-width
            # sum(Y^2) reduction doesn't fold garbage in. No-op for full bands.
            for band_idx in range(len(column_bands)):
                bp_lo = column_bands[band_idx][0]
                band_real_free = (column_bands[band_idx][2] - column_bands[band_idx][1]) * w_tile_size
                if band_real_free < effective_free_dim:
                    nisa.memset(
                        dst=psum_tile[bp_lo : bp_lo + c_out_tile_size, band_real_free:effective_free_dim],
                        value=0.0,
                        engine=_get_memset_engine(band_idx, memset_engine_modulo),
                    )

            # Fused copy + bias + sum(Y): result = psum + bias (or 0); free-dim reduction is
            # RMW-accumulated into the persistent per-channel sum column.
            operand0 = bias_sbuf if has_bias else 0.0
            nisa.tensor_scalar_reduce(
                dst=result,
                data=psum_data,
                op0=nl.add,
                operand0=operand0,
                reduce_op=nl.add,
                reduce_res=sum_col,
                reduce_cmd=nisa.reduce_cmd.load_reduce,
                reduce_init=sum_col,
            )

            # sum(Y^2): square the result (throwaway dst), reduce along free dim into this flush's
            # partials column; cross-flush accumulation deferred to the finalize pass.
            nisa.activation(
                dst=bn_sq_scratch[:evict_p, :effective_free_dim],
                op=nl.square,
                data=result,
                reduce_op=nl.add,
                reduce_res=partial_sqsum,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )
        else:
            if has_bias and has_activation:
                nisa.tensor_scalar(dst=result, data=psum_data, op0=nl.add, operand0=bias_sbuf)
                nisa.activation(dst=result, data=result, op=get_nl_act_fn_from_type(activation_fn))
            elif has_bias:
                nisa.tensor_scalar(dst=result, data=psum_data, op0=nl.add, operand0=bias_sbuf)
            elif has_activation:
                nisa.activation(dst=result, data=psum_data, op=get_nl_act_fn_from_type(activation_fn))
            else:
                nisa.tensor_copy(dst=result, src=psum_data)

            # Fill never-written free positions in the result SBUF (memset targets SBUF, not PSUM)
            # with f(0 + bias), per band in band-local free coords.
            if has_unwritten:
                _fill_band_free_gaps_sbuf(
                    result_full,
                    bias_sbuf,
                    column_bands,
                    c_out_tile_size,
                    unwritten_free_ranges,
                    effective_free_dim,
                    w_tile_size,
                    has_bias,
                    has_activation,
                    activation_fn,
                    memset_engine_modulo,
                )


def _load_input_window_to_sbuf_3d(
    x_in_cin: nl.NkiTensor,
    input_window_buf: nl.NkiTensor,
    cfg: Conv3dConfig,
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
) -> tuple[Optional[nl.NkiTensor], int, int, int, int, int, int]:
    """
    Load the union input window from HBM into an SBUF buffer via DMA.

    Computes the bounding box of all receptive fields for the given D-H output
    positions and W tile range, clips to valid input bounds, and issues a single
    DMA copy. Returns None if the entire window falls in padding.
    """
    c_in_tile_size = x_in_cin.shape[0]

    union_d_start, union_d_end = cfg.D, 0
    union_h_start, union_h_end = cfg.H, 0
    for dh_idx in range(len(dh_positions)):
        d_out, h_out = dh_positions[dh_idx]
        field_d_start = d_out * cfg.stride_d - cfg.pad_d_left
        field_d_end = d_out * cfg.stride_d + (cfg.K_d - 1) * cfg.dilation_d - cfg.pad_d_left + 1
        field_h_start = h_out * cfg.stride_h - cfg.pad_h_top
        field_h_end = h_out * cfg.stride_h + (cfg.K_h - 1) * cfg.dilation_h - cfg.pad_h_top + 1
        union_d_start = min(union_d_start, field_d_start)
        union_d_end = max(union_d_end, field_d_end)
        union_h_start = min(union_h_start, field_h_start)
        union_h_end = max(union_h_end, field_h_end)

    field_w_start = w_start * cfg.stride_w - cfg.pad_w_left
    field_w_end = (w_end - 1) * cfg.stride_w + (cfg.K_w - 1) * cfg.dilation_w - cfg.pad_w_left + 1

    valid_d_start, valid_d_end = max(0, union_d_start), min(cfg.D, union_d_end)
    valid_h_start, valid_h_end = max(0, union_h_start), min(cfg.H, union_h_end)
    valid_w_start, valid_w_end = max(0, field_w_start), min(cfg.W, field_w_end)

    if valid_d_start >= valid_d_end or valid_h_start >= valid_h_end or valid_w_start >= valid_w_end:
        return None, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end

    d_window_size = valid_d_end - valid_d_start
    h_window_size = valid_h_end - valid_h_start
    w_window_size = valid_w_end - valid_w_start

    nisa.dma_copy(
        dst=input_window_buf[:c_in_tile_size, :d_window_size, :h_window_size, :w_window_size],
        src=x_in_cin.slice(dim=1, start=valid_d_start, end=valid_d_end, step=1)
        .slice(dim=2, start=valid_h_start, end=valid_h_end, step=1)
        .slice(dim=3, start=valid_w_start, end=valid_w_end, step=1),
        dge_mode=nisa.dge_mode.hwdge,
        engine=nki.isa.engine.sync,
    )
    return input_window_buf, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end


def _conv3d_output_tile(
    x_in_batch: nl.NkiTensor,
    filters_cache: list[list[list[nl.NkiTensor]]],
    bias_cache: list[Optional[nl.NkiTensor]],
    result_sbufs: list[nl.NkiTensor],
    input_window_slots: list[nl.NkiTensor],
    stacked_input_slots: list[list[nl.NkiTensor]],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_tile_sizes: list[int],
    dh_positions: list[tuple[int, int]],
    w_start: int,
    w_end: int,
    psum_bank_idx: int,
    c_in_slot_offset: int = 0,
    preloaded_windows: Optional[list[tuple]] = None,
    bn_c_out_tile_base: int = 0,
    bn_sum_acc: Optional[nl.NkiTensor] = None,
    bn_sq_scratch: Optional[nl.NkiTensor] = None,
    bn_sqsum_partials: Optional[nl.NkiTensor] = None,
    bn_partial_iters: int = 1,
    bn_iter_idx: int = 0,
) -> None:
    """
    Compute one output tile by iterating over C_in tiles with multi-buffered I/O.

    When the number of c_out tiles exceeds NUM_PSUM_BANKS, processes them in
    sub-groups that fit within the available PSUM banks. For each sub-group:
    iterates over all C_in tiles (load input, scatter, matmul), then flushes
    PSUM results to SBUF before reusing the banks for the next sub-group.
    """
    num_c_out_tiles = len(result_sbufs)
    w_tile_size = w_end - w_start
    num_dh_positions = len(dh_positions)

    # Column tiling packs col_rep bands of band_dh positions (band_dh == whole flush at col_rep 1;
    # a ragged tail keeps band_dh == dh_head_size with the last band partial).
    col_rep, band_dh, col_size = _flush_column_layout(tile_cfg, num_dh_positions)
    per_band_free = band_dh * w_tile_size  # per-band free width fed to PSUM / eviction
    effective_free_dim = num_dh_positions * w_tile_size  # total packed D-H positions this flush

    # Fully-padded free positions get no matmul (flush-global coords).
    unwritten_free_ranges = _compute_unwritten_free_ranges(cfg, tile_cfg, dh_positions, w_start, w_end)

    # Bands (partition_offset, dh_lo, dh_hi); single band at col_rep == 1.
    column_bands = _column_bands(num_dh_positions, band_dh, col_rep, col_size)
    if col_rep > 1:
        # One PSUM bank, col_rep * col_size partitions tall x per_band_free wide (<= F_MAX).
        psum_part_dim = col_rep * col_size
        psum_free_dim = per_band_free
        evict_partitions = col_rep * col_size
    else:
        psum_part_dim = None  # per-tile size below
        psum_free_dim = per_band_free
        evict_partitions = 0

    # Process c_out tiles in sub-groups that fit within PSUM banks
    for sub_group_start in range(0, num_c_out_tiles, _NUM_PSUM_BANKS):
        sub_group_end = min(sub_group_start + _NUM_PSUM_BANKS, num_c_out_tiles)
        sub_group_size = sub_group_end - sub_group_start
        sub_c_out_tile_sizes = c_out_tile_sizes[sub_group_start:sub_group_end]

        # Allocate PSUM banks for this sub-group
        psum_tiles = []
        for sg_idx in range(sub_group_size):
            c_out_tile_idx = sub_group_start + sg_idx
            part_dim = psum_part_dim if psum_part_dim != None else c_out_tile_sizes[c_out_tile_idx]
            psum_tiles.append(
                nl.ndarray(
                    shape=(part_dim, psum_free_dim),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, (psum_bank_idx + sg_idx) % _NUM_PSUM_BANKS * _PSUM_BANK_SIZE),
                )
            )

        # Iterate over all C_in tiles for this sub-group
        c_in_tile_idx = 0
        for c_in_start in range(0, cfg.C_in, tile_cfg.P_MAX):
            c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
            c_in_tile_size = c_in_end - c_in_start

            # Multi-buffer slot selection
            global_idx = c_in_slot_offset + c_in_tile_idx
            input_window_buf = input_window_slots[global_idx % len(input_window_slots)]
            stacked_input_bufs = stacked_input_slots[global_idx % len(stacked_input_slots)]

            if preloaded_windows != None:
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    preloaded_windows[c_in_tile_idx]
                )
            else:
                x_in_cin = x_in_batch.slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                input_window, valid_d_start, valid_d_end, valid_h_start, valid_h_end, valid_w_start, valid_w_end = (
                    _load_input_window_to_sbuf_3d(
                        x_in_cin=x_in_cin,
                        input_window_buf=input_window_buf,
                        cfg=cfg,
                        dh_positions=dh_positions,
                        w_start=w_start,
                        w_end=w_end,
                    )
                )

            # Get filters for this sub-group
            filters_for_cin = []
            for sg_idx in range(sub_group_size):
                c_out_tile_idx = sub_group_start + sg_idx
                filters_for_cin.append(filters_cache[c_out_tile_idx][c_in_tile_idx])

            _conv3d_cin_tile(
                input_window=input_window,
                filters_for_cin=filters_for_cin,
                psum_tiles=psum_tiles,
                stacked_input_bufs=stacked_input_bufs,
                cfg=cfg,
                c_in_tile_size=c_in_tile_size,
                dh_positions=dh_positions,
                w_start=w_start,
                w_end=w_end,
                valid_d_start=valid_d_start,
                valid_d_end=valid_d_end,
                valid_h_start=valid_h_start,
                valid_h_end=valid_h_end,
                valid_w_start=valid_w_start,
                valid_w_end=valid_w_end,
                used_k_positions=tile_cfg.used_k_positions,
                tensor_copy_engine_idx=c_in_tile_idx * tile_cfg.K_total_used,
                effective_free_dim=effective_free_dim,
                tensor_copy_engine_modulo=tile_cfg.tensor_copy_activation_engine_offload,
                memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
                c_out_tile_sizes=sub_c_out_tile_sizes,
                C_OUT_REP=col_rep,
                col_size=col_size,
                dh_head_size=band_dh,
            )
            c_in_tile_idx += 1

        # Flush this sub-group's PSUM results to SBUF
        sub_result_sbufs = result_sbufs[sub_group_start:sub_group_end]
        sub_bias_sbufs = bias_cache[sub_group_start:sub_group_end]
        _apply_bias_activation_and_copy(
            psum_tiles=psum_tiles,
            result_sbufs=sub_result_sbufs,
            bias_sbufs=sub_bias_sbufs,
            c_out_tile_sizes=sub_c_out_tile_sizes,
            has_bias=cfg.has_bias,
            # Batchnorm defers activation to the rescale phase, so stats see raw conv output.
            has_activation=cfg.has_activation and not cfg.fuse_batch_norm,
            activation_fn=cfg.activation_fn,
            # Eviction is per band: each band is per_band_free wide, evict_partitions spans all bands.
            effective_free_dim=per_band_free,
            unwritten_free_ranges=unwritten_free_ranges,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
            fuse_batch_norm=cfg.fuse_batch_norm,
            bn_sum_acc=bn_sum_acc,
            bn_sq_scratch=bn_sq_scratch,
            bn_sqsum_partials=bn_sqsum_partials,
            bn_partial_iters=bn_partial_iters,
            bn_iter_idx=bn_iter_idx,
            bn_tile_offset=bn_c_out_tile_base + sub_group_start,
            evict_partitions=evict_partitions,
            column_bands=column_bands,
            w_tile_size=w_tile_size,
        )


# Memory Hierarchy 1: HBM <-> SBUF


def _allocate_filter_bias_buffers(
    sbm: SbufManager,
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    mem_cfg: Conv3dMemoryConfig,
    dtype,
) -> tuple[list[Optional[nl.NkiTensor]], list[list[nl.NkiTensor]]]:
    """
    Allocate SBUF heap buffers for bias and filter storage.

    Creates interleaved bias buffers (one per C_out tile in the interleave group)
    and filter buffers indexed by [c_in_tile][k_outer_tile].
    """
    c_out_wide = mem_cfg.c_out_interleave * tile_cfg.c_out_tile_size_max

    # Column tiling replicates bias across the C_OUT_REP bands (free dim 1, no extra per-partition
    # SBUF) so each band's eviction reads its per-channel bias.
    bias_part_dim = tile_cfg.C_OUT_REP * tile_cfg.col_size if tile_cfg.C_OUT_REP > 1 else tile_cfg.c_out_tile_size_max
    bias_bufs: list[Optional[nl.NkiTensor]] = []
    for tile_idx in range(mem_cfg.c_out_interleave):
        if cfg.has_bias:
            bias_bufs.append(
                sbm.alloc_heap(
                    shape=(bias_part_dim, 1),
                    dtype=nl.float32,
                    name=f"bias_{tile_idx}",
                )
            )
        else:
            bias_bufs.append(None)

    filter_bufs: list[list[nl.NkiTensor]] = []
    for cin_idx in range(tile_cfg.c_in_tile_count):
        cin_filters: list[nl.NkiTensor] = []
        for k_idx in range(tile_cfg.K_outer_tile_count):
            cin_filters.append(
                sbm.alloc_heap(
                    shape=(tile_cfg.stacked_filter_dim_max, c_out_wide),
                    dtype=dtype,
                    name=f"filter_cin{cin_idx}_k{k_idx}",
                )
            )
        cin_filters_list = cin_filters
        filter_bufs.append(cin_filters_list)

    return bias_bufs, filter_bufs


def _load_bias_into_buf(
    bias: nl.NkiTensor,
    bias_buf: nl.NkiTensor,
    c_out_start: int,
    c_out_end: int,
    C_OUT_REP: int = 1,
    col_size: int = 128,
) -> nl.NkiTensor:
    """
    Load a slice of the bias tensor from HBM into an SBUF buffer via DMA.

    Under column tiling (C_OUT_REP > 1) the bias is replicated into each of the C_OUT_REP
    partition bands (offsets band_idx * col_size), so the eviction of a band reads its per-channel
    bias directly. The band offsets are multiples of col_size (32 or 64), i.e. quadrant
    aligned, so a plain tensor_copy suffices. Returns the full (possibly replicated) buffer.
    """
    c_out_size = c_out_end - c_out_start
    nisa.dma_copy(
        dst=bias_buf[:c_out_size, 0],
        src=bias.slice(dim=0, start=c_out_start, end=c_out_end, step=1),
    )
    if C_OUT_REP > 1:
        for band_idx in range(1, C_OUT_REP):
            nisa.tensor_copy(
                dst=bias_buf[band_idx * col_size : band_idx * col_size + c_out_size, :],
                src=bias_buf[0:c_out_size, :],
            )
        return bias_buf
    return bias_buf.slice(dim=0, start=0, end=c_out_size, step=1)


def _load_filters_into_bufs(
    filters: nl.NkiTensor,
    filter_bufs: list[nl.NkiTensor],
    cfg: Conv3dConfig,
    c_in_start: int,
    c_in_end: int,
    c_out_group_start: int,
    c_out_group_end: int,
    used_k_positions: list[int],
    memset_engine_idx: int = 0,
    memset_engine_modulo: int = 2,
) -> None:
    """
    Load filter weights from HBM into K-replicated stacked SBUF buffers.

    For each K-outer tile, packs multiple filter positions along the partition
    dimension with the appropriate partition stride. Zero-initializes buffers
    when the partition stride exceeds the C_in tile size to handle padding.
    """
    K_h, K_w = cfg.K_h, cfg.K_w
    K_total_used = len(used_k_positions)
    c_in_size = c_in_end - c_in_start
    c_out_wide = c_out_group_end - c_out_group_start
    K_REP, partition_stride = _get_k_replication_params(c_in_size, K_total_used)
    K_outer_tile_count = div_ceil(K_total_used, K_REP)

    for k_outer_idx in range(K_outer_tile_count):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K_total_used)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual
        raw_buf = filter_bufs[k_outer_idx]

        if partition_stride > c_in_size:
            nisa.memset(
                dst=raw_buf[:stacked_filter_dim, :c_out_wide],
                value=0.0,
                engine=_get_memset_engine(memset_engine_idx + k_outer_idx, memset_engine_modulo),
            )

        for k_rep_idx in range(k_actual):
            partition_offset = k_rep_idx * partition_stride
            flat_k_pos = used_k_positions[k_start + k_rep_idx]
            k_d_idx, k_h_idx, k_w_idx = _decompose_k_position(flat_k_pos, K_h, K_w)
            nisa.dma_copy(
                dst=raw_buf[partition_offset : partition_offset + c_in_size, :c_out_wide],
                src=filters.select(dim=0, index=k_d_idx)
                .select(dim=0, index=k_h_idx)
                .select(dim=0, index=k_w_idx)
                .slice(dim=0, start=c_in_start, end=c_in_end, step=1)
                .slice(dim=1, start=c_out_group_start, end=c_out_group_start + c_out_wide, step=1),
                dge_mode=nisa.dge_mode.hwdge,
            )


def _load_bias_and_filters_for_c_out_group(
    filters: nl.NkiTensor,
    bias: Optional[nl.NkiTensor],
    cfg: Conv3dConfig,
    tile_cfg: Conv3dTileConfig,
    c_out_group_start: int,
    c_out_group_end: int,
    pre_bias_bufs: list[Optional[nl.NkiTensor]],
    pre_filter_bufs: list[list[nl.NkiTensor]],
) -> tuple[list[int], list[Optional[nl.NkiTensor]], list[list[list[nl.NkiTensor]]]]:
    """
    Load bias and filter weights for an entire C_out interleave group.

    Loads bias per c_out tile, then loads all filter weights for the entire
    c_out group. Returns sliced views for each c_out tile.
    """
    actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)
    c_out_tile_sizes = []
    bias_cache: list[Optional[nl.NkiTensor]] = []

    for c_out_tile_idx in range(actual_group_size):
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = min(c_out_tile_start + tile_cfg.P_MAX, tile_cfg.C_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        c_out_tile_sizes.append(c_out_tile_size)

        if cfg.has_bias:
            bias_cache.append(
                _load_bias_into_buf(
                    bias,
                    pre_bias_bufs[c_out_tile_idx],
                    c_out_tile_start,
                    c_out_tile_end,
                    C_OUT_REP=tile_cfg.C_OUT_REP,
                    col_size=tile_cfg.col_size,
                )
            )
        else:
            bias_cache.append(None)

    K_total_used = len(tile_cfg.used_k_positions)

    for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
        c_in_start = c_in_tile_idx * tile_cfg.P_MAX
        c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
        _load_filters_into_bufs(
            filters,
            pre_filter_bufs[c_in_tile_idx],
            cfg,
            c_in_start,
            c_in_end,
            c_out_group_start,
            c_out_group_end,
            used_k_positions=tile_cfg.used_k_positions,
            memset_engine_modulo=tile_cfg.memset_gpsimd_engine_offload,
        )

    filters_cache: list[list[list[nl.NkiTensor]]] = []
    for c_out_tile_idx in range(actual_group_size):
        c_out_offset = c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_size = c_out_tile_sizes[c_out_tile_idx]
        tile_filters: list[list[nl.NkiTensor]] = []
        for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
            c_in_size = min(tile_cfg.P_MAX, cfg.C_in - c_in_tile_idx * tile_cfg.P_MAX)
            K_REP_cin, partition_stride_cin = _get_k_replication_params(c_in_size, K_total_used)
            K_outer_tile_count_cin = div_ceil(K_total_used, K_REP_cin)
            cin_k_views: list[nl.NkiTensor] = []
            for k_outer_idx in range(K_outer_tile_count_cin):
                k_start = k_outer_idx * K_REP_cin
                k_end = min(k_start + K_REP_cin, K_total_used)
                k_actual = k_end - k_start
                stacked_filter_dim = partition_stride_cin * k_actual
                wide_buf = pre_filter_bufs[c_in_tile_idx][k_outer_idx]
                cin_k_views.append(wide_buf[:stacked_filter_dim, c_out_offset : c_out_offset + c_out_tile_size])
            tile_filters.append(cin_k_views)
        filters_cache.append(tile_filters)

    return c_out_tile_sizes, bias_cache, filters_cache
