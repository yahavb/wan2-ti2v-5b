"""Neuron-compatible RoPE implementation for Wan2.2.

Replaces the complex-arithmetic rope_apply with real-valued cos/sin math.
Neuron doesn't support: float64, complex types, view_as_complex, torch.polar.

This provides:
  - rope_params_neuron: returns (freqs_cos, freqs_sin) instead of complex freqs
  - rope_apply_neuron: uses real-valued rotation instead of complex multiplication
"""
import torch


def rope_params_neuron(max_seq_len, dim, theta=10000):
    """Compute rotary position embedding frequencies as real cos/sin.
    
    Replaces the original rope_params which returns complex polar freqs.
    Returns (freqs_cos, freqs_sin) each of shape [max_seq_len, dim//2].
    """
    # Compute angles (same as original but stay in float32)
    angles = torch.outer(
        torch.arange(max_seq_len).float(),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).float().div(dim)))
    return torch.cos(angles), torch.sin(angles)


def rope_apply_neuron(x, grid_sizes, freqs):
    """Apply rotary position embeddings using real-valued math.
    
    Equivalent to the original rope_apply but avoids:
    - torch.view_as_complex / torch.view_as_real
    - float64 / complex128
    - torch.polar
    
    Args:
        x: [B, L, N, D] query or key tensor
        grid_sizes: [B, 3] tensor with (f, h, w) per sample
        freqs: tuple (freqs_cos, freqs_sin) each [max_seq_len, D//2]
    
    Returns: [B, L, N, D] with RoPE applied (bf16)
    """
    n, c = x.size(2), x.size(3) // 2  # num_heads, half_head_dim
    
    # Split frequency dims: (c - 2*(c//3), c//3, c//3) for (frame, height, width)
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    
    freqs_cos, freqs_sin = freqs
    freqs_cos_split = freqs_cos.split([s0, s1, s1], dim=1)
    freqs_sin_split = freqs_sin.split([s0, s1, s1], dim=1)
    
    # Loop over batch
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        
        # Build cos/sin grids: [seq_len, 1, c]
        cos_grid = torch.cat([
            freqs_cos_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_cos_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_cos_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, c)
        
        sin_grid = torch.cat([
            freqs_sin_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_sin_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_sin_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, c)
        
        # Extract x for this sample: [seq_len, N, D]
        x_seq = x[i, :seq_len].float()
        
        # Reshape to pairs: [seq_len, N, c, 2] where c = D//2
        x_pairs = x_seq.reshape(seq_len, n, c, 2)
        x_re = x_pairs[..., 0]  # [seq_len, N, c]
        x_im = x_pairs[..., 1]  # [seq_len, N, c]
        
        # Apply rotation: (x_re + i*x_im) * (cos + i*sin)
        # Real part: x_re * cos - x_im * sin
        # Imag part: x_re * sin + x_im * cos
        out_re = x_re * cos_grid - x_im * sin_grid
        out_im = x_re * sin_grid + x_im * cos_grid
        
        # Interleave back: [seq_len, N, c, 2] -> [seq_len, N, D]
        out = torch.stack([out_re, out_im], dim=-1).flatten(2)
        
        # Concat with any remaining tokens beyond seq_len
        x_i = torch.cat([out, x[i, seq_len:].float()])
        output.append(x_i)
    
    return torch.stack(output).to(x.dtype)
