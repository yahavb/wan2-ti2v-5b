# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os

import torch
import contextlib  # patched for Neuron
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─── NKI Kernel Loading for Neuron ─────────────────────────────────────
_USE_NKI_VAE = os.environ.get("USE_NKI_VAE", "1") == "1"
# VAE_COMPILE=1: decode through models/vae_static.py (fixed-shape causal cache, ported from
# vllm-omni-neuron) under torch.compile: one graph for chunk 0, one for every later chunk.
_VAE_COMPILE = os.environ.get("VAE_COMPILE", "0") == "1"
# VAE_FUSE_NORM=1: each RMS_norm + SiLU pair runs as one torch.compile'd graph (1 kernel instead
# of ~9 eager ones, one HBM read + write instead of several fp32 passes).
_VAE_FUSE_NORM = os.environ.get("VAE_FUSE_NORM", "0") == "1"
# VAE_SUBPIXEL=1 (with USE_NKI_CONV3D=1): each upsample conv (nearest-2x + 3x3) runs as four 2x2
# convs on the low-res frames, one per output phase, plus one interleave copy: 2.25x fewer MACs
# and no 4x upsampled input. Differs from the original by the bf16 rounding of the summed taps.
_VAE_SUBPIXEL = os.environ.get("VAE_SUBPIXEL", "0") == "1"
# VAE_FUSE_CAT=1 (with VAE_FUSE_NORM=1, USE_NKI_CONV3D=1): the compiled RMS_norm + SiLU in front of
# each nkilib-eligible causal conv also writes its output behind the 2 cached frames and returns
# the next chunk's cache, replacing the eager torch.cat (a full copy of the conv input), the
# cache .clone() and the time pad. Same values.
_VAE_FUSE_CAT = os.environ.get("VAE_FUSE_CAT", "0") == "1"
# VAE_FUSE_RES=1 (with VAE_FUSE_CAT=1): each ResidualBlock's last conv adds the shortcut inside the
# nkilib kernel (residual_add_loc) instead of a separate full-size eager add. One bf16 rounding of
# conv + shortcut instead of two.
_VAE_FUSE_RES = os.environ.get("VAE_FUSE_RES", "0") == "1"
# VAE_FULLGRAPH=1: fullgraph=True, so any graph break fails loudly instead of silently
# splitting the decoder into fragments.
_VAE_FULLGRAPH = os.environ.get("VAE_FULLGRAPH", "0") == "1"
_nki_conv2d_k1 = None
_nki_conv2d_k3 = None
_nki_self_attn = None
_NKI_AVAILABLE = False

if _USE_NKI_VAE:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.vae_conv2d import vae_conv2d_k1, vae_conv2d_k3_shifted
        from kernels.vae_attention import vae_self_attention
        _nki_conv2d_k1 = wrap_nki(vae_conv2d_k1)
        _nki_conv2d_k3 = wrap_nki(vae_conv2d_k3_shifted)
        _nki_self_attn = wrap_nki(vae_self_attention)
        _NKI_AVAILABLE = True
        logging.info("[vae2_2] NKI VAE kernels: LOADED")
    except Exception as e:
        logging.warning(f"[vae2_2] NKI VAE kernels: FAILED ({e}), using PyTorch fallback")

# USE_NKI_CONV3D=1: causal 3x3x3 convs with C_in, C_out >= 96 and D_out >= 2 run the nkilib
# conv3d kernel (kernels/vae_conv3d.py). NKI_CONV3D_CHECK=1 logs, once per shape, the error
# against the eager conv.
_USE_NKI_CONV3D = os.environ.get("USE_NKI_CONV3D", "0") == "1"
_NKI_CONV3D_CHECK = os.environ.get("NKI_CONV3D_CHECK", "0") == "1"
_nki_conv3d = None
_nki_conv3d_checked = set()
if _USE_NKI_CONV3D:
    try:
        from kernels import vae_conv3d as _nki_conv3d
        logging.info("[vae2_2] nkilib conv3d: LOADED")
    except Exception as e:
        logging.warning(f"[vae2_2] nkilib conv3d: FAILED ({e}), using PyTorch conv3d")


def _is_neuron_tensor(x):
    """Check if tensor is on a Neuron device."""
    return x.device.type == "neuron" or (hasattr(x.device, 'type') and 'xla' in str(x.device))


def _nki_conv2d_forward(weight, bias, x_2d, kernel_size, C_in, C_out, H, W, padding=0):
    """Run spatial conv2d via NKI kernel. x_2d is (BT, C, H, W) squeezed to 2D."""
    BT = x_2d.shape[0]
    device = x_2d.device
    P = 128
    SPATIAL_TILE = 512

    results = []
    for bt in range(BT):
        frame = x_2d[bt]  # (C_in, H, W)
        HW = H * W

        if kernel_size == 1:
            # Flatten to (C_in, HW), pad
            inp_flat = frame.reshape(C_in, HW).to(torch.bfloat16)
            C_in_p = ((C_in + P - 1) // P) * P
            C_out_p = ((C_out + P - 1) // P) * P
            HW_p = ((HW + SPATIAL_TILE - 1) // SPATIAL_TILE) * SPATIAL_TILE

            inp_padded = torch.zeros(C_in_p, HW_p, dtype=torch.bfloat16, device=device)
            inp_padded[:C_in, :HW] = inp_flat

            # Weight_T: (C_in, C_out) transposed
            w = weight.reshape(C_out, C_in).to(torch.bfloat16)
            w_T = torch.zeros(C_in_p, C_out_p, dtype=torch.bfloat16, device=device)
            w_T[:C_in, :C_out] = w.T

            b_padded = torch.zeros(C_out_p, 1, dtype=torch.bfloat16, device=device)
            if bias is not None:
                b_padded[:C_out, 0] = bias.to(torch.bfloat16)

            out = _nki_conv2d_k1(inp_padded, w_T, b_padded, HW)
            results.append(out[:C_out, :HW].reshape(C_out, H, W))

        elif kernel_size == 3:
            inp_flat = frame.reshape(C_in, HW).to(torch.bfloat16)
            HW_p = ((HW + SPATIAL_TILE - 1) // SPATIAL_TILE) * SPATIAL_TILE
            C_in_p = ((C_in + P - 1) // P) * P
            C_out_p = ((C_out + P - 1) // P) * P

            # Build shifted inputs
            frame_3d = frame.reshape(C_in, H, W)
            x_padded = F.pad(frame_3d.float(), (padding, padding, padding, padding))
            shifts = []
            for kh in range(3):
                for kw in range(3):
                    window = x_padded[:, kh:kh + H, kw:kw + W]
                    shifts.append(window.reshape(C_in, HW))
            shifted = torch.stack(shifts, dim=0).reshape(9 * C_in, HW).to(torch.bfloat16)

            # Pad channels and spatial
            shifted_padded = torch.zeros(9 * C_in_p, HW_p, dtype=torch.bfloat16, device=device)
            for k_idx in range(9):
                shifted_padded[k_idx * C_in_p:k_idx * C_in_p + C_in, :HW] = \
                    shifted[k_idx * C_in:k_idx * C_in + C_in, :HW]

            # Weight_T
            w_4d = weight.reshape(C_out, C_in, 3, 3)
            w_T_padded = torch.zeros(C_in_p * 9, C_out_p, dtype=torch.bfloat16, device=device)
            for k_idx in range(9):
                kh, kw = k_idx // 3, k_idx % 3
                w_slice = w_4d[:, :, kh, kw].T.to(torch.bfloat16)  # (C_in, C_out)
                w_T_padded[k_idx * C_in_p:k_idx * C_in_p + C_in, :C_out] = w_slice

            b_padded = torch.zeros(C_out_p, 1, dtype=torch.bfloat16, device=device)
            if bias is not None:
                b_padded[:C_out, 0] = bias.to(torch.bfloat16)

            out = _nki_conv2d_k3(shifted_padded, w_T_padded, b_padded, HW)
            results.append(out[:C_out, :HW].reshape(C_out, H, W))

    return torch.stack(results, dim=0)  # (BT, C_out, H, W)

__all__ = [
    "Wan2_2_VAE",
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)
        self._nki_filters = None  # packed lazily, after the model is on device

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        # Once packed (first call with D_out >= 2), every call takes the kernel: the original
        # weight is freed, and the D_out=1 calls left are chunk 0's.
        if self._nki_filters is not None or (
                _nki_conv3d is not None and x.shape[2] + padding[4] - 2 >= _nki_conv3d.MIN_D_OUT
                and _nki_conv3d.eligible(self)):
            return self._nki_forward(x, padding)
        x = F.pad(x, padding)

        return super().forward(x)

    def forward_time_padded(self, z, residual=None):
        """forward() for an input that already carries the 2 causal frames in front
        (VAE_FUSE_CAT); same kernel choice and result. residual (VAE_FUSE_RES) is added to the
        output, inside the kernel on the nkilib path."""
        if self._nki_filters is not None or (
                _nki_conv3d is not None and z.shape[2] - 2 >= _nki_conv3d.MIN_D_OUT
                and _nki_conv3d.eligible(self)):
            return self._nki_forward(z, [0, 0, 0, 0, 0, 0], residual)
        p = self._padding
        out = super().forward(F.pad(z, (p[0], p[1], p[2], p[3], 0, 0)))
        return out if residual is None else out + residual

    def _nki_forward(self, x, padding, residual=None):
        if padding[4] > 0:  # fewer than 2 cached frames: zero-pad time, kernel pads H/W
            x = F.pad(x, (0, 0, 0, 0, padding[4], 0))
        if self._nki_filters is None:
            self._nki_filters = _nki_conv3d.pack_filters(self.weight)
            if not _NKI_CONV3D_CHECK:  # drop the unpacked copy (~0.5 GB per rank in total)
                self.weight = nn.Parameter(self.weight.new_empty(0), requires_grad=False)
        out = _nki_conv3d.conv3d_3x3x3(x.to(self._nki_filters.dtype), self._nki_filters, self.bias,
                                       residual)
        if _NKI_CONV3D_CHECK:
            key = (tuple(x.shape), self.out_channels, residual is not None)
            if key not in _nki_conv3d_checked:
                _nki_conv3d_checked.add(key)
                ref = super().forward(F.pad(x, (1, 1, 1, 1, 0, 0)))
                if residual is not None:
                    ref = ref + residual
                err = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
                logging.warning(f"[nki_conv3d] x={list(x.shape)} C_out={self.out_channels} "
                                f"rel_err={err.item():.4f} shape_ok={out.shape == ref.shape}")
        return out


def _rms_silu_impl(x, gamma):
    """RMS_norm(channel_first, bias=False) followed by SiLU, as one compiled graph."""
    rms = x.to(torch.float32).pow(2).mean(1, keepdim=True).clamp(min=1e-8).rsqrt()
    return F.silu((x * rms).to(gamma.dtype) * gamma)


def _rms_silu_cat_impl(x, gamma, prev):
    """_rms_silu written behind the 2 cached frames prev: (conv input [b, C, 2 + T, H, W],
    the next chunk's 2-frame cache)."""
    z = torch.cat([prev, _rms_silu_impl(x, gamma)], dim=2)
    return z, z[:, :, -CACHE_T:].clone()


def _rms_silu_pad_impl(x, gamma):
    """First chunk: _rms_silu behind 2 zero frames (the causal padding)."""
    z = F.pad(_rms_silu_impl(x, gamma), (0, 0, 0, 0, CACHE_T, 0))
    return z, z[:, :, -CACHE_T:].clone()


_rms_silu = None
_rms_silu_cat = None
_rms_silu_pad = None
if _VAE_FUSE_NORM:
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    _rms_silu = torch.compile(_rms_silu_impl, backend="neuron", dynamic=False)
    if _VAE_FUSE_CAT:
        _rms_silu_cat = torch.compile(_rms_silu_cat_impl, backend="neuron", dynamic=False)
        _rms_silu_pad = torch.compile(_rms_silu_pad_impl, backend="neuron", dynamic=False)


def _fused_cat_conv(layers, i):
    """VAE_FUSE_CAT: (conv, index) of the nkilib-eligible CausalConv3d fed by the fusable
    RMS_norm + SiLU at layers[i] (an eval-mode Dropout in between is skipped), else (None, None)."""
    if _rms_silu_cat is None or _nki_conv3d is None:
        return None, None
    j = i + 2
    if j < len(layers) and isinstance(layers[j], nn.Dropout) and not layers[j].training:
        j += 1
    if j < len(layers) and isinstance(layers[j], CausalConv3d) and _nki_conv3d.eligible(layers[j]):
        return layers[j], j
    return None, None


def _norm_cat_conv(x, gamma, conv, feat_cache, idx, residual=None):
    """RMS_norm + SiLU + causal conv with the cache concat done inside the compiled norm graph.
    Same values as the eager path: the conv sees [2 cached (or zero) frames, normed x] and the
    cache keeps the last 2 frames of that (a zero frame stands in for a missing one).
    residual: added to the conv output (inside the nkilib kernel)."""
    prev = feat_cache[idx]
    if prev is None:
        z, cache = _rms_silu_pad(x, gamma)
    else:
        if prev.shape[2] < CACHE_T:
            prev = F.pad(prev, (0, 0, 0, 0, CACHE_T - prev.shape[2], 0))
        z, cache = _rms_silu_cat(x, gamma, prev)
    feat_cache[idx] = cache
    return conv.forward_time_padded(z, residual)


def _norm_silu_fusable(layers, i):
    """layers[i] is a bias-free channel-first RMS_norm followed by SiLU (fusable pair)."""
    return (_rms_silu is not None and i + 1 < len(layers) and isinstance(layers[i], RMS_norm)
            and isinstance(layers[i + 1], nn.SiLU) and layers[i].channel_first
            and not isinstance(layers[i].bias, torch.Tensor))


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        # x * rsqrt(mean(x^2)) == F.normalize(x) * sqrt(D), without F.normalize's
        # NaN-guard and generic L2 reduce (vllm-omni measured ~67us/call on Neuron).
        dim = 1 if self.channel_first else -1
        rms = x.to(torch.float32).pow(2).mean(dim, keepdim=True).clamp(min=1e-8).rsqrt()
        return (x * rms).to(self.gamma.dtype) * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        # Integer 2x nearest == repeat each pixel; avoids nn.Upsample's fp32 cast and the
        # indirect-DMA gathers it lowers to on Neuron (vllm-omni-neuron-staging).
        if tuple(self.scale_factor) == (2.0, 2.0) and self.mode in ("nearest", "nearest-exact"):
            return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return super().forward(x.float()).type_as(x)


class NkiConv2d(nn.Conv2d):
    """3x3 / pad 1 Conv2d run as nkilib conv3d with D=1 when USE_NKI_CONV3D=1 (the
    upsample convs; staging measured them paying off together with the 3D group)."""

    def forward(self, x):
        if _nki_conv3d is None or min(self.in_channels, self.out_channels) < _nki_conv3d.MIN_CHANNELS:
            return super().forward(x)
        if getattr(self, "_nki_filters", None) is None:
            self._nki_filters = _nki_conv3d.pack_filters(self.weight.unsqueeze(2))
            if not _NKI_CONV3D_CHECK:
                self.weight = nn.Parameter(self.weight.new_empty(0), requires_grad=False)
        out = _nki_conv3d.conv3d_3x3x3(x.unsqueeze(2).to(self._nki_filters.dtype),
                                       self._nki_filters, self.bias).squeeze(2)
        if _NKI_CONV3D_CHECK and (tuple(x.shape), "2d") not in _nki_conv3d_checked:
            _nki_conv3d_checked.add((tuple(x.shape), "2d"))
            ref = super().forward(x)
            err = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
            logging.warning(f"[nki_conv2d] x={list(x.shape)} rel_err={err.item():.4f}")
        return out


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, dim, 3, padding=1),
                # nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] != "Rep"):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                    cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] == "Rep"):
                        cache_x = torch.cat(
                            [
                                torch.zeros_like(cache_x).to(cache_x.device),
                                cache_x
                            ],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        if (_VAE_SUBPIXEL and self.mode in ("upsample2d", "upsample3d") and _nki_conv3d is not None
                and _is_neuron_tensor(x) and self.dim >= _nki_conv3d.MIN_CHANNELS):
            return self._subpixel(x)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def _subpixel(self, x):
        """VAE_SUBPIXEL: nearest-2x + 3x3 conv on [b, c, t, h, w] as 4 phase 2x2 convs on the
        low-res frames (kernels/vae_conv3d.upsample_conv_subpixel); no (b t) rearranges."""
        conv = self.resample[1]
        if getattr(self, "_phase_filters", None) is None:
            self._phase_filters = _nki_conv3d.pack_phase_filters(conv.weight)
            if not _NKI_CONV3D_CHECK:
                conv.weight = nn.Parameter(conv.weight.new_empty(0), requires_grad=False)
        out = _nki_conv3d.upsample_conv_subpixel(x, self._phase_filters, conv.bias)
        if _NKI_CONV3D_CHECK and (tuple(x.shape), "subpix") not in _nki_conv3d_checked:
            _nki_conv3d_checked.add((tuple(x.shape), "subpix"))
            t = x.shape[2]
            ref = F.conv2d(self.resample[0](rearrange(x, "b c t h w -> (b t) c h w")),
                           conv.weight, conv.bias, padding=1)
            ref = rearrange(ref, "(b t) c h w -> b c t h w", t=t)
            err = (out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)
            logging.warning(f"[subpixel] x={list(x.shape)} rel_err={err.item():.4f}")
        return out

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity())

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        layers = list(self.residual)
        skip_to = -1
        for i, layer in enumerate(layers):
            if i <= skip_to:  # already applied by a fused norm (SiLU, and with VAE_FUSE_CAT the conv)
                continue
            if _norm_silu_fusable(layers, i):
                conv, j = _fused_cat_conv(layers, i) if feat_cache is not None else (None, None)
                if conv is not None:
                    # VAE_FUSE_RES: the block's last conv adds the shortcut inside the kernel
                    res = h if (_VAE_FUSE_RES and _nki_conv3d.HAS_RESIDUAL
                                and j == len(layers) - 1) else None
                    x = _norm_cat_conv(x, layer.gamma, conv, feat_cache, feat_idx[0], res)
                    feat_idx[0] += 1
                    skip_to = j
                    if res is not None:
                        return x
                else:
                    x = _rms_silu(x, layer.gamma)
                    skip_to = i + 1
                continue
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)

        # Use NKI kernels if available and on Neuron
        if _NKI_AVAILABLE and _is_neuron_tensor(x):
            # NKI path: use vae_conv2d_k1 for QKV/proj, vae_self_attention for SDPA
            P = 128
            CHUNK = 512
            seq = h * w
            seq_padded = ((seq + CHUNK - 1) // CHUNK) * CHUNK

            # QKV via NKI conv2d_k1
            qkv = _nki_conv2d_forward(
                self.to_qkv.weight, self.to_qkv.bias,
                x, kernel_size=1, C_in=c, C_out=c * 3, H=h, W=w)
            # qkv: (BT, 3*c, h, w)

            results = []
            for bt_idx in range(b * t):
                qkv_frame = qkv[bt_idx]  # (3*c, h, w)
                qkv_flat = qkv_frame.reshape(3 * c, seq)
                q_flat, k_flat, v_flat = qkv_flat.chunk(3, dim=0)  # each (c, seq)

                # NKI attention expects: q(1,d,seq), k(1,d,seq), v(1,seq,d)
                q_nki = q_flat.unsqueeze(0).to(torch.bfloat16)  # (1, c, seq)
                k_nki = k_flat.unsqueeze(0).to(torch.bfloat16)  # (1, c, seq)
                v_nki = v_flat.T.unsqueeze(0).to(torch.bfloat16)  # (1, seq, c)

                # Pad seq to multiple of 512
                if seq < seq_padded:
                    q_nki = F.pad(q_nki, (0, seq_padded - seq))
                    k_nki = F.pad(k_nki, (0, seq_padded - seq))
                    v_nki = F.pad(v_nki, (0, 0, 0, seq_padded - seq))

                identity_mat = torch.eye(P, dtype=torch.bfloat16, device=x.device)
                scale = 1.0 / math.sqrt(c)

                out_nki = _nki_self_attn(q_nki, k_nki, v_nki, identity_mat, softmax_scale=scale)
                # out_nki: (seq_padded, 1, c) → trim to (seq, c)
                out_frame = out_nki[:seq, 0, :]  # (seq, c)
                out_frame = out_frame.T.reshape(c, h, w)  # (c, h, w)
                results.append(out_frame)

            x = torch.stack(results, dim=0)  # (BT, c, h, w)

            # Proj via NKI conv2d_k1
            x = _nki_conv2d_forward(
                self.proj.weight, self.proj.bias,
                x, kernel_size=1, C_in=c, C_out=c, H=h, W=w)
        else:
            # PyTorch fallback
            q, k, v = (
                self.to_qkv(x).reshape(b * t, 1, c * 3,
                                       -1).permute(0, 1, 3,
                                                   2).contiguous().chunk(3, dim=-1))
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
            x = self.proj(x)

        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_downsample=False,
                 down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_upsample=False,
                 up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i]
                if i < len(temperal_downsample) else False)
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                ))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(
                temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                ))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        head_layers = list(self.head)
        skip_to = -1
        for i, layer in enumerate(head_layers):
            if i <= skip_to:  # already applied by the fused norm
                continue
            if _norm_silu_fusable(head_layers, i):
                conv, j = _fused_cat_conv(head_layers, i) if feat_cache is not None else (None, None)
                if conv is not None:
                    x = _norm_cat_conv(x, layer.gamma, conv, feat_cache, feat_idx[0])
                    feat_idx[0] += 1
                    skip_to = j
                else:
                    x = _rms_silu(x, layer.gamma)
                    skip_to = i + 1
                continue
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        # Cast back to model dtype after scale division (avoids float32 vs bf16 mismatch in conv2)
        z = z.to(self.conv2.weight.dtype)
        if _VAE_COMPILE:
            return self._decode_compiled(z)
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def decode_chunks(self, z, scale):
        """Eager decode() that yields each latent frame's output as soon as it is decoded
        ([1, 3, f, H, W]; f = 1 for the first chunk, 4 after). Same ops as decode(), so the
        concatenated chunks equal decode()'s output."""
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        z = z.to(self.conv2.weight.dtype)
        x = self.conv2(z)
        for i in range(z.shape[2]):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map,
                                feat_idx=self._conv_idx, first_chunk=(i == 0))
            yield unpatchify(out_, patch_size=2)
        self.clear_cache()

    def _decode_compiled(self, z):
        """VAE_COMPILE path: static-cache decoder, compiled once (2 graphs)."""
        from models.vae_static import StaticCacheDecoder, decode_static
        first = getattr(self, "_static_decoder", None) is None
        if first:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._static_decoder = torch.compile(
                StaticCacheDecoder(self), backend="neuron", dynamic=False, fullgraph=_VAE_FULLGRAPH)
            from torch._dynamo.utils import counters
            graphs0 = counters["stats"].get("unique_graphs", 0)
            breaks0 = sum(counters["graph_break"].values())
            print(f"[vae2_2] compiling static-cache decoder: fullgraph={_VAE_FULLGRAPH} "
                  f"NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '')!r}", flush=True)
            import time as _time
            t0 = _time.time()
        # z is already scaled by decode(); pass identity scale.
        out = decode_static(self, z, [0.0, 1.0], self._static_decoder)
        if first:
            from torch._dynamo.utils import counters
            print(f"[vae2_2] first compiled decode {_time.time() - t0:.1f}s; graphs compiled: "
                  f"{counters['stats'].get('unique_graphs', 0) - graphs0}, graph breaks: "
                  f"{sum(counters['graph_break'].values()) - breaks0} {dict(counters['graph_break']) or ''}",
                  flush=True)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", **kwargs):
    # params
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class Wan2_2_VAE:

    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float,
        device="cuda",
    ):

        self.dtype = dtype
        self.device = device

        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            ).eval().requires_grad_(False).to(device))

    def encode(self, videos):
        try:
            if not isinstance(videos, list):
                raise TypeError("videos should be a list")
            with contextlib.nullcontext():  # autocast removed for Neuron
                return [
                    self.model.encode(u.unsqueeze(0),
                                      self.scale).float().squeeze(0)
                    for u in videos
                ]
        except TypeError as e:
            logging.info(e)
            return None

    def decode_chunks(self, z):
        """Streaming decode of one latent [C, T, H, W]: yields [3, f, H*16, W*16] float32
        clamped chunks (the same values decode([z])[0] has for those frames)."""
        for out in self.model.decode_chunks(z.unsqueeze(0), self.scale):
            yield out.float().clamp_(-1, 1).squeeze(0)

    def decode(self, zs):
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            with contextlib.nullcontext():  # autocast removed for Neuron
                return [
                    self.model.decode(u.unsqueeze(0),
                                      self.scale).float().clamp_(-1,
                                                                 1).squeeze(0)
                    for u in zs
                ]
        except TypeError as e:
            logging.info(e)
            return None
