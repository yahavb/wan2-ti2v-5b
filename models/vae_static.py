"""Compile-friendly Wan2.2 VAE decode: fixed-shape causal cache, two graphs.

Port of the vllm-omni-neuron approach (NeuronWanDecoder3d / _init_feat_cache) onto
our Wan2.2 (48-channel, residual) VAE in wan/modules/vae2_2.py. The original decoder
keeps its causal-conv cache in a Python list whose entries go None -> "Rep" ->
1-frame tensor -> 2-frame tensor across chunks; under torch.compile that means
sentinel guards, list-mutation side effects and a different graph per cache shape.

Here every cache entry is a zero tensor of shape [B, C_in, 2, h, w] from the start
(zero frames == the causal zero padding the original applies), passed into the graph
as flat inputs and returned as outputs. The update for every causal conv is
    new_cache = cat(old_cache, x)[:, :, -2:]
which reproduces the original prepend/slice rules exactly. `first_chunk` replaces
the "Rep" sentinel: on chunk 0 the temporal upsample conv is skipped (its cache stays
zero). Result: one graph for chunk 0 and one for every later chunk.

The modules are the original ones (same weights); only the cache plumbing differs.
"""

import torch
import torch.nn as nn

from wan.modules.vae2_2 import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    Resample,
    ResidualBlock,
    rearrange,
    unpatchify,
)


def _cconv(conv, x, cache, idx):
    """Causal conv with a fixed 2-frame cache; updates cache[idx] in place."""
    i = idx[0]
    old = cache[i]
    y = conv(x, old)
    cache[i] = torch.cat([old, x], dim=2)[:, :, -CACHE_T:].contiguous()
    idx[0] += 1
    return y


def _resblock(blk, x, cache, idx):
    h = blk.shortcut(x)
    for layer in blk.residual:
        x = _cconv(layer, x, cache, idx) if isinstance(layer, CausalConv3d) else layer(x)
    return x + h


def _resample(rs, x, cache, idx, first_chunk):
    """Decoder-side Resample (upsample2d / upsample3d)."""
    b, c, t, h, w = x.shape
    if rs.mode == "upsample3d":
        if first_chunk:
            idx[0] += 1  # original: cache set to "Rep", time_conv skipped
        else:
            y = _cconv(rs.time_conv, x, cache, idx)
            y = y.reshape(b, 2, c, t, h, w)
            x = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = rs.resample(x)
    return rearrange(x, "(b t) c h w -> b c t h w", t=t)


def _upblock(ub, x, cache, idx, first_chunk):
    xm = x
    for m in ub.upsamples:
        if isinstance(m, ResidualBlock):
            xm = _resblock(m, xm, cache, idx)
        elif isinstance(m, Resample):
            xm = _resample(m, xm, cache, idx, first_chunk)
        else:
            xm = m(xm)
    if ub.avg_shortcut is not None:
        return xm + ub.avg_shortcut(x, first_chunk)
    return xm


class StaticCacheDecoder(nn.Module):
    """conv2 + Decoder3d for ONE latent frame; cache in/out as flat tensors."""

    def __init__(self, vae):  # vae: WanVAE_
        super().__init__()
        self.conv2 = vae.conv2
        self.decoder = vae.decoder

    def forward(self, z_frame, *flat_cache, first_chunk: bool):
        dec = self.decoder
        cache = list(flat_cache)
        idx = [0]
        x = self.conv2(z_frame)  # 1x1x1 causal conv: per-frame == whole-latent
        x = _cconv(dec.conv1, x, cache, idx)
        for layer in dec.middle:
            x = _resblock(layer, x, cache, idx) if isinstance(layer, ResidualBlock) else layer(x)
        for ub in dec.upsamples:
            x = _upblock(ub, x, cache, idx, first_chunk)
        for layer in dec.head:
            x = _cconv(layer, x, cache, idx) if isinstance(layer, CausalConv3d) else layer(x)
        assert idx[0] == len(cache), f"cache slots used {idx[0]} != allocated {len(cache)}"
        return (x, *cache)


def init_cache(vae, z):
    """Zero cache tensors in decoder execution order. z: [B, z_dim, T, H, W] latent."""
    dec = vae.decoder
    B, _, _, H, W = z.shape
    cache = []

    def add(conv, h, w):
        cache.append(torch.zeros(B, conv.in_channels, CACHE_T, h, w, device=z.device, dtype=z.dtype))

    def add_res(blk, h, w):
        for layer in blk.residual:
            if isinstance(layer, CausalConv3d):
                add(layer, h, w)

    add(dec.conv1, H, W)
    for layer in dec.middle:
        if isinstance(layer, ResidualBlock):
            add_res(layer, H, W)
    for ub in dec.upsamples:
        for m in ub.upsamples:
            if isinstance(m, ResidualBlock):
                add_res(m, H, W)
            elif isinstance(m, Resample):
                if m.mode == "upsample3d":
                    add(m.time_conv, H, W)
                if m.mode in ("upsample3d", "upsample2d"):
                    H, W = H * 2, W * 2
    for layer in dec.head:
        if isinstance(layer, CausalConv3d):
            add(layer, H, W)
    return cache


def decode_static(vae, z, scale, decoder_fn):
    """Mirror of WanVAE_.decode using StaticCacheDecoder (compiled or not)."""
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, vae.z_dim, 1, 1, 1) + scale[0].view(1, vae.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    z = z.to(vae.conv2.weight.dtype)
    cache = init_cache(vae, z)
    outs = []
    for i in range(z.shape[2]):
        res = decoder_fn(z[:, :, i:i + 1], *cache, first_chunk=(i == 0))
        outs.append(res[0])
        cache = list(res[1:])
    return unpatchify(torch.cat(outs, dim=2), patch_size=2)


__all__ = ["StaticCacheDecoder", "decode_static", "init_cache"]
