"""Neuron-compatible RoPE implementation for Wan2.2.

Adapted from rolling-forcing/app/models/layers.py.
Neuron doesn't support: float64, complex types, view_as_complex, torch.polar,
tensor select (x[i]), torch.stack on device.

This provides:
  - rope_params_neuron: returns angles tensor [max_seq_len, dim//2] (for torch.cat compatibility)
  - rope_apply_neuron: builds cos/sin grids on CPU, transfers to device, applies rotation
"""
import torch


def rope_params_neuron(max_seq_len, dim, theta=10000):
    """Compute rotary position embedding frequency angles.
    
    Returns a [max_seq_len, dim//2] float32 tensor of angles.
    Compatible with torch.cat along dim=1 (same as original rope_params shape).
    """
    angles = torch.outer(
        torch.arange(max_seq_len).float(),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).float().div(dim)))
    return angles


def rope_apply_neuron(x, grid_sizes, freqs):
    """Apply rotary position embeddings using real-valued math.
    
    Adapted from rolling-forcing causal_rope_apply. Key Neuron constraints:
    - No tensor select (x[i]) — use slice x[i:i+1] + reshape
    - No torch.stack — use torch.cat with unsqueeze
    - No float64/complex — stay in float32
    - Build cos/sin grids on CPU (Neuron compiler can't handle cat of expanded views)
    
    Args:
        x: [B, L, N, D] query or key tensor (on Neuron device)
        grid_sizes: [B, 3] tensor with (f, h, w) per sample
        freqs: [max_seq_len, D//2] angles tensor (on Neuron device after model.to())
    
    Returns: [B, L, N, D] with RoPE applied (same dtype as input)
    """
    n, c = x.size(2), x.size(3) // 2  # num_heads, half_head_dim
    
    # Split frequency dims: (c - 2*(c//3), c//3, c//3) for (frame, height, width)
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    
    # Move freqs to CPU for grid construction (Neuron can't handle cat of expanded views)
    freqs_cpu = freqs.cpu().float()
    freqs_split = freqs_cpu.split([s0, s1, s1], dim=1)
    
    device = x.device
    
    # Loop over batch (usually B=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        
        # Build angle grids on CPU: [seq_len, 1, c]
        angles_grid = torch.cat([
            freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, c)
        
        # Compute cos/sin on CPU then transfer to device
        cos_grid = torch.cos(angles_grid).to(device)
        sin_grid = torch.sin(angles_grid).to(device)
        
        # Extract x for this sample using slice (not select — unsupported on Neuron)
        # x[i:i+1, :seq_len] gives [1, seq_len, N, D]
        x_seq = x[i:i+1, :seq_len].to(torch.float32)  # [1, seq_len, N, D]
        
        # Reshape to pairs: [1, seq_len, N, c, 2]
        x_pairs = x_seq.reshape(1, seq_len, n, c, 2)
        # Use slice 0:1/1:2 instead of [..., 0] select (unsupported on Neuron)
        x_re = x_pairs[:, :, :, :, 0:1].reshape(1, seq_len, n, c)
        x_im = x_pairs[:, :, :, :, 1:2].reshape(1, seq_len, n, c)
        
        # Apply rotation: (x_re + i*x_im) * (cos + i*sin)
        out_re = x_re * cos_grid - x_im * sin_grid
        out_im = x_re * sin_grid + x_im * cos_grid
        
        # Interleave with unsqueeze+cat (not stack — unsupported on Neuron)
        x_out = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], dim=-1)
        x_out = x_out.reshape(1, seq_len, n, c * 2)  # [1, seq_len, N, D]
        
        # Concat with remaining tokens beyond seq_len
        x_rest = x[i:i+1, seq_len:].to(torch.float32)
        x_i = torch.cat([x_out, x_rest], dim=1)  # [1, L, N, D]
        output.append(x_i)
    
    # Cat along batch dim (not stack)
    result = torch.cat(output, dim=0)  # [B, L, N, D]
    return result.to(x.dtype)
