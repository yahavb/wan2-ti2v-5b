"""nkilib conv3d for the VAE's causal 3x3x3 convs (port of vllm-omni-neuron-staging's dispatch).

CausalConv3d prepends the cached frames (and, when fewer than 2 are cached, zero frames), so the
conv the kernel runs always has padding (d_l, d_r, h_t, h_b, w_l, w_r) = (0, 0, 1, 1, 1, 1): one
traced config covers every site. Filters are packed once to [K_d, K_h, K_w, C_in, C_out].

Gates (from staging's full-model A/B): C_in and C_out >= 96 and D_out >= 2. Below these the
compiler/eager conv wins; D_out=1 regressed vae_decode by 25% there.
"""
import nki
import torch

try:
    from nkilib.experimental.conv.conv3d import conv3d
except ImportError:
    from kernels.nkilib_vendor.experimental.conv.conv3d import conv3d
from torch_neuronx.nki_hop import wrap_nki

PADDING = (0, 0, 1, 1, 1, 1)
MIN_CHANNELS = 96
MIN_D_OUT = 2


@nki.jit
def _conv3d_3x3x3_kernel(x, filters, bias):
    return conv3d(x, filters, bias, stride=(1, 1, 1), padding=PADDING, dilation=(1, 1, 1))


_conv3d_3x3x3 = wrap_nki(_conv3d_3x3x3_kernel)


def eligible(conv):
    """Static (weight-only) gate for a CausalConv3d."""
    return (tuple(conv.kernel_size) == (3, 3, 3) and tuple(conv.stride) == (1, 1, 1)
            and tuple(conv.dilation) == (1, 1, 1) and conv.groups == 1 and conv.bias is not None
            and conv._padding[0] == 1 and conv._padding[2] == 1
            and conv.in_channels >= MIN_CHANNELS and conv.out_channels >= MIN_CHANNELS)


def pack_filters(weight):
    """[C_out, C_in, K_d, K_h, K_w] -> [K_d, K_h, K_w, C_in, C_out]; permute on CPU (a permuted
    device tensor can't be made contiguous in place)."""
    return weight.detach().cpu().permute(2, 3, 4, 1, 0).contiguous().to(weight.device)


def conv3d_3x3x3(x, filters, bias):
    """x: [B, C_in, D, H, W] with the causal time padding already applied."""
    return _conv3d_3x3x3(x.contiguous(), filters, bias)
