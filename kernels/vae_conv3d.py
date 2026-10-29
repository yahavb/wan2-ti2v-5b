"""nkilib conv3d for the VAE's causal 3x3x3 convs (port of vllm-omni-neuron-staging's dispatch).

CausalConv3d prepends the cached frames (and, when fewer than 2 are cached, zero frames), so the
conv the kernel runs always has padding (d_l, d_r, h_t, h_b, w_l, w_r) = (0, 0, 1, 1, 1, 1): one
traced config covers every site. Filters are packed once to [K_d, K_h, K_w, C_in, C_out].

Gates (from staging's full-model A/B): C_in and C_out >= 96 and D_out >= 2. Below these the
compiler/eager conv wins; D_out=1 regressed vae_decode by 25% there.
"""
import os

import nki
import torch

try:
    from nkilib.experimental.conv.conv3d import conv3d
except ImportError:
    from kernels.nkilib_vendor.experimental.conv.conv3d import conv3d
try:  # fused residual add (installed nkilib; the vendored copy predates it)
    from nkilib.experimental.conv.conv3d import ResidualAddLoc
except ImportError:
    ResidualAddLoc = None
from torch_neuronx.nki_hop import wrap_nki

PADDING = (0, 0, 1, 1, 1, 1)
MIN_CHANNELS = 96
MIN_D_OUT = 2
# NKI_CONV3D_SMALL_COUT=1: also take convs with a small C_out (C_in still >= MIN_CHANNELS) -- in
# the decoder that is only the head conv, 256 -> 12 at full tile resolution.
SMALL_COUT = os.environ.get("NKI_CONV3D_SMALL_COUT", "0") == "1"


@nki.jit
def _conv3d_3x3x3_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1))


_conv3d_3x3x3 = wrap_nki(_conv3d_3x3x3_kernel)
# VAE_STRIPS (exact strips with halo rows): the input already carries the neighbours' rows, so
# the kernel pads only W.
PADDING_HP0 = (0, 0, 0, 0, 1, 1)
@nki.jit
def _conv3d_3x3x3_hp0_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING_HP0, dilation=(1, 1, 1))
_conv3d_3x3x3_hp0 = wrap_nki(_conv3d_3x3x3_hp0_kernel)
HAS_RESIDUAL = ResidualAddLoc is not None


@nki.jit
def _conv3d_3x3x3_res_kernel(x, filters, bias, residual):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1),
                  residual_add_loc=ResidualAddLoc.POST_ACT, residuals_in=residual)


_conv3d_3x3x3_res = wrap_nki(_conv3d_3x3x3_res_kernel) if HAS_RESIDUAL else None
@nki.jit
def _conv3d_3x3x3_res_hp0_kernel(x, filters, bias, residual):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING_HP0, dilation=(1, 1, 1),
                  residual_add_loc=ResidualAddLoc.POST_ACT, residuals_in=residual)
_conv3d_3x3x3_res_hp0 = wrap_nki(_conv3d_3x3x3_res_hp0_kernel) if HAS_RESIDUAL else None


def eligible(conv):
    """Static (weight-only) gate for a CausalConv3d."""
    return (tuple(conv.kernel_size) == (3, 3, 3) and tuple(conv.stride) == (1, 1, 1)
            and tuple(conv.dilation) == (1, 1, 1) and conv.groups == 1 and conv.bias is not None
            and conv._padding[0] == 1 and conv._padding[2] == 1
            and conv.in_channels >= MIN_CHANNELS
            and (conv.out_channels >= MIN_CHANNELS or SMALL_COUT))


# Sub-pixel form of nearest-2x upsample + 3x3/pad-1 conv: output phase (py, px), i.e. pixels
# (2i + py, 2j + px), is a 2x2 conv on the low-res input with H padding (top, bottom) = (1, 0)
# for py = 0 and (0, 1) for py = 1 (W likewise for px), using the 3x3 taps summed per phase.
@nki.jit
def _phase00_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 1, 0, 1, 0), dilation=(1, 1, 1))


@nki.jit
def _phase01_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 1, 0, 0, 1), dilation=(1, 1, 1))


@nki.jit
def _phase10_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 0, 1, 1, 0), dilation=(1, 1, 1))


@nki.jit
def _phase11_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 0, 1, 0, 1), dilation=(1, 1, 1))


_PHASES = [wrap_nki(k) for k in (_phase00_kernel, _phase01_kernel, _phase10_kernel, _phase11_kernel)]
# Halo'd input (one extra low-res row above and below): every phase pads W only; phase py reads
# input rows [py, py + h + 1) of the h + 2.
@nki.jit
def _phase_w10_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 0, 0, 1, 0), dilation=(1, 1, 1))
@nki.jit
def _phase_w01_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=(0, 0, 0, 0, 0, 1), dilation=(1, 1, 1))
_PHASES_HP0 = [wrap_nki(k) for k in (_phase_w10_kernel, _phase_w01_kernel)]  # by px
# rows of the 3 taps folded into each of the 2 taps of a phase (index [phase][tap2][tap3])
_FOLD = ([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def pack_phase_filters(weight):
    """3x3 Conv2d weight [C_out, C_in, 3, 3] -> 4 packed 2x2 phase filters [1, 2, 2, C_in, C_out]
    (phase index 2*py + px). Taps are summed in fp32 and rounded to the weight dtype once."""
    w3 = weight.detach().cpu().float()
    out = []
    for py in (0, 1):
        for px in (0, 1):
            ry, rx = torch.tensor(_FOLD[py]), torch.tensor(_FOLD[px])
            w2 = torch.einsum("ak,oikl,bl->oiab", ry, w3, rx).to(weight.dtype)
            out.append(w2.unsqueeze(2).permute(2, 3, 4, 1, 0).contiguous().to(weight.device))
    return out


def upsample_conv_subpixel(x, phase_filters, bias, halo_rows=None, phases_only=False):
    """x [B, C_in, T, h, w] -> nearest-2x + 3x3 conv output [B, C_out, T, 2h, 2w]: one 2x2 conv
    per phase on the low-res frames (2.25x fewer MACs, no 4x upsampled input), then one
    interleave copy. halo_rows (VAE_STRIPS): (rows -1 .. h-1, rows 0 .. h) of x with the
    neighbours' rows, [B, C_in, T, h + 1, w] each; the phases then pad W only. phases_only: return
    the 4 phase outputs [B, C_out, T, h, w] (index 2*py + px) without interleaving them."""
    B, _, T, h, w = x.shape
    if halo_rows is None:
        x = x.to(phase_filters[0].dtype).contiguous()
        ys = [_PHASES[p](x, phase_filters[p], bias) for p in range(4)]  # 4 x [B, C_out, T, h, w]
    else:
        rows = [r.to(phase_filters[0].dtype).contiguous() for r in halo_rows]  # by py
        ys = [_PHASES_HP0[p % 2](rows[p // 2], phase_filters[p], bias) for p in range(4)]
    if phases_only:
        return ys
    c_out = ys[0].shape[1]
    y = torch.stack(ys, dim=2).view(B, c_out, 2, 2, T, h, w)       # [B, C, py, px, T, h, w]
    return y.permute(0, 1, 4, 5, 2, 6, 3).reshape(B, c_out, T, 2 * h, 2 * w)


def pack_filters(weight):
    """[C_out, C_in, K_d, K_h, K_w] -> [K_d, K_h, K_w, C_in, C_out]; permute on CPU (a permuted
    device tensor can't be made contiguous in place)."""
    return weight.detach().cpu().permute(2, 3, 4, 1, 0).contiguous().to(weight.device)


def conv3d_3x3x3(x, filters, bias, residual=None, hpad=True):
    """x: [B, C_in, D, H, W] with the causal time padding already applied. residual (optional,
    [B, C_out, D - 2, H, W], same dtype): added to the output inside the kernel. hpad=False: x
    already carries one halo row above and below (output H = input H - 2)."""
    if residual is not None:
        k = _conv3d_3x3x3_res if hpad else _conv3d_3x3x3_res_hp0
        return k(x.contiguous(), filters, bias, residual.to(x.dtype).contiguous())
    return (_conv3d_3x3x3 if hpad else _conv3d_3x3x3_hp0)(x.contiguous(), filters, bias)
