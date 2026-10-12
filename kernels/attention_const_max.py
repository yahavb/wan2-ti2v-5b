"""Local (no all-gather) constant-max self-attention for TP-only DiT.

Reuses the compute core of kernels/attention_const_max_allgather.py (odyssey-trainium):
softmax(Q @ K.T * scale - c) @ V with a constant max c, no online-softmax max reduction
and no P transpose. Here all of K/V is local, so there is one KV chunk and no collective.

Layouts (same as the AG kernel's pre-transposed chunks):
    q   [N, d, Sq]  bf16
    k_t [N, d, Sk]  bf16  (pre-transposed K)
    v   [N, Sk, d]  bf16
    out [N, Sq, d]  bf16
d must be 128. Work is sharded across LNC cores on heads when they divide evenly, else on Sq.
"""

import os

import nki
import nki.language as nl

from kernels.attention_const_max_allgather import (
    _KV_TILES_PER_BLOCK,
    _KVBlockStream,
    _StreamWrapper,
    _attention_const_max_compute,
    _prepare_softmax_max,
)
from nkilib.core.utils.kernel_assert import kernel_assert

# NEURON_LOGICAL_NC_CONFIG if set, else this repo's LNC_CONFIG label (LNC1/LNC2), else 2.
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG")
          or os.environ.get("LNC_CONFIG", "LNC2").upper().replace("LNC", "") or 2)


class _LocalStream(_StreamWrapper):
    """Plain (non-transposing) stream; the compute loop toggles a store only the AG K uses."""

    def set_store_enabled(self, enabled: bool) -> None:
        pass


@nki.jit
def attention_const_max_local(q, k_t, v, softmax_max, softmax_scale):
    Nq, d, Sq = q.shape
    kernel_assert(d == 128, f"d must be 128, got {d}")
    o = nl.ndarray(shape=(Nq, Sq, d), dtype=q.dtype, buffer=nl.shared_hbm)

    num_cores = nl.num_programs()
    core_id = nl.program_id(0)
    if Nq % num_cores == 0:
        hpc = Nq // num_cores
        h0 = core_id * hpc
        q_l, o_l = q[h0:h0 + hpc, :, :], o[h0:h0 + hpc, :, :]
        k_l, v_l = k_t[h0:h0 + hpc, :, :], v[h0:h0 + hpc, :, :]
    else:
        sq = Sq // num_cores
        s0 = core_id * sq
        s1 = Sq if core_id == num_cores - 1 else s0 + sq
        q_l, o_l = q[:, :, s0:s1], o[:, s0:s1, :]
        k_l, v_l = k_t, v

    n_local = q_l.shape[0]
    k_streams = [[_LocalStream(_KVBlockStream(
        k_l[h], block_size=(1, _KV_TILES_PER_BLOCK), tiling_dim=1, transpose=False))]
        for h in range(n_local)]
    v_streams = [[_LocalStream(_KVBlockStream(
        v_l[h], block_size=(_KV_TILES_PER_BLOCK, 1), tiling_dim=0))]
        for h in range(n_local)]

    _attention_const_max_compute(
        q_l, k_streams, v_streams, o_l,
        softmax_max=_prepare_softmax_max(softmax_max),
        softmax_scale=softmax_scale,
        collective_callbacks_by_head=None,
    )
    return o


# Launch with one program per physical core of the logical core (LNC2 -> 2).
attention_const_max = attention_const_max_local[LNC]
