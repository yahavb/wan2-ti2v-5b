"""Test suite for VAE NKI kernels — conv2d_k1, conv2d_k3_shifted, vae_self_attention.

Run on Neuron instance:
    NEURON_RT_NUM_CORES=2 python test_vae_kernels.py

Tests each kernel against a CPU PyTorch reference implementation.
Follows the exact same pattern as test_all_kernels.py from rolling-forcing.

Production shapes from VAE decoder (Wan2.2, 480p, 81 frames):
  Decoder stages (dec_dim=256, dim_mult=[1,2,4,4]):
    Stage 0 (bottleneck): C=1024, H=30, W=52   (1560 spatial tokens)
    Stage 1 (after up2d):  C=1024, H=60, W=104  (6240 spatial tokens)
    Stage 2 (after up3d):  C=512,  H=120, W=208 (24960 spatial tokens)
    Stage 3 (no upsample): C=256,  H=240, W=416 (99840 spatial tokens)
"""
import os
import sys
import math
import traceback

if "NEURON_RT_NUM_CORES" not in os.environ:
    os.environ["NEURON_RT_NUM_CORES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "kernels"))

DEVICE = torch.device("neuron")
TOL = 0.05  # bf16 tolerance

results = []


def record(name, max_diff, err=None):
    ok = err is None and max_diff is not None and max_diff < TOL
    status = "PASS" if ok else "FAIL"
    if err:
        msg = f"  [{status}] {name:<60s}  ERROR: {err[:120]}"
    else:
        msg = f"  [{status}] {name:<60s}  max_diff = {max_diff:.6f}"
    print(msg)
    results.append((name, ok, max_diff, err))


# ════════════════════════════════════════════════════════════════════════
# Helper: build shifted inputs for conv2d_k3
# ════════════════════════════════════════════════════════════════════════
def build_shifted_inputs(x_2d, H, W, K=3, padding=1):
    """Given input (C, H*W), pad and extract 9 shifted views for 3×3 conv.

    Returns (9*C, HW_padded) where each C-channel block is one kernel position.
    """
    C = x_2d.shape[0]
    # Reshape to (C, H, W)
    x_3d = x_2d.reshape(C, H, W)
    # Pad spatially
    x_padded = F.pad(x_3d, (padding, padding, padding, padding))  # (C, H+2, W+2)
    H_pad, W_pad = x_padded.shape[1], x_padded.shape[2]

    shifts = []
    for kh in range(K):
        for kw in range(K):
            # Extract window: x_padded[:, kh:kh+H, kw:kw+W]
            window = x_padded[:, kh:kh + H, kw:kw + W]  # (C, H, W)
            shifts.append(window.reshape(C, H * W))

    # Stack: (9, C, HW) → reshape to (9*C, HW)
    stacked = torch.stack(shifts, dim=0)  # (9, C, HW)
    result = stacked.reshape(9 * C, H * W)

    # Pad HW to multiple of 512
    HW = H * W
    pad_hw = (512 - HW % 512) % 512
    if pad_hw > 0:
        result = F.pad(result, (0, pad_hw))

    return result


def build_weight_slices(weight_4d):
    """Reshape Conv2d weight (C_out, C_in, 3, 3) to (C_out, C_in*9) in blocked layout.

    Layout: col = k_idx * C_in + c_in where k_idx = kh*3 + kw
    """
    C_out, C_in, K, K2 = weight_4d.shape
    assert K == 3 and K2 == 3
    # weight_4d[:, :, kh, kw] → slice for kernel position kh*3+kw
    slices = []
    for kh in range(K):
        for kw in range(K):
            slices.append(weight_4d[:, :, kh, kw])  # (C_out, C_in)
    # Concat along dim=1: (C_out, C_in*9)
    return torch.cat(slices, dim=1)


# ════════════════════════════════════════════════════════════════════════
# TEST: Conv2d K=1 (pointwise)
# ════════════════════════════════════════════════════════════════════════
def test_conv2d_k1():
    print("\n── Conv2d K=1 (vae_conv2d_k1) ───────────────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_conv2d import vae_conv2d_k1
    except Exception as e:
        record("conv2d_k1: import", None, f"{type(e).__name__}: {e}")
        return

    # Production shapes from VAE decoder
    # (name, C_in, C_out, H, W)
    configs = [
        # conv1 in decoder: z_dim → bottleneck
        ("dec_conv1_48→1024_30x52",    128, 1024, 30, 52),   # pad 48→128
        # conv2 in WanVAE_: z_dim pointwise
        ("dec_conv2_128→128_30x52",    128, 128,  30, 52),
        # Shortcut conv 1×1 in ResBlock (when in_dim != out_dim)
        ("resblock_1024→512_120x208",  1024, 512,  120, 208),
        ("resblock_512→256_240x416",   512,  256,  240, 416),
        # QKV conv in AttentionBlock
        ("attn_qkv_1024→3072_30x52",  1024, 3072, 30, 52),   # pad 3072 → 3072 (ok, multiple of 128)
        # Proj conv in AttentionBlock
        ("attn_proj_1024→1024_30x52",  1024, 1024, 30, 52),
    ]

    wrapped = wrap_nki(vae_conv2d_k1)

    for name, C_in, C_out, H, W in configs:
        try:
            torch.manual_seed(0)
            # Ensure C_in and C_out are multiples of 128
            C_in_padded = ((C_in + 127) // 128) * 128
            C_out_padded = ((C_out + 127) // 128) * 128

            HW = H * W
            HW_padded = ((HW + 511) // 512) * 512

            # Create PyTorch Conv2d reference
            conv = nn.Conv2d(C_in, C_out, 1, bias=True)
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)

            # Input
            x = torch.randn(1, C_in, H, W, dtype=torch.float32)

            # CPU reference
            with torch.no_grad():
                ref = conv(x)  # (1, C_out, H, W)
            ref_flat = ref.reshape(C_out, HW).to(torch.bfloat16)

            # Prepare NKI inputs
            x_flat = x.reshape(C_in, HW).to(torch.bfloat16)
            # Pad channels to multiples of 128
            if C_in < C_in_padded:
                x_flat = F.pad(x_flat, (0, 0, 0, C_in_padded - C_in))
            # Pad spatial to multiple of 512
            if HW < HW_padded:
                x_flat = F.pad(x_flat, (0, HW_padded - HW))

            # Weight: (C_out, C_in) → pad to (C_out_padded, C_in_padded)
            w = conv.weight.data.reshape(C_out, C_in).to(torch.bfloat16)
            w_padded = torch.zeros(C_out_padded, C_in_padded, dtype=torch.bfloat16)
            w_padded[:C_out, :C_in] = w

            # Bias: (C_out,) → (C_out_padded, 1)
            b = conv.bias.data.to(torch.bfloat16)
            b_padded = torch.zeros(C_out_padded, 1, dtype=torch.bfloat16)
            b_padded[:C_out, 0] = b

            # Run NKI kernel
            out = wrapped(
                x_flat.to(DEVICE),
                w_padded.to(DEVICE),
                b_padded.to(DEVICE),
                HW
            ).cpu()

            # Trim to actual size
            out = out[:C_out, :HW]

            diff = (out.float() - ref_flat.float()).abs().max().item()
            record(f"conv2d_k1: {name} (Ci={C_in},Co={C_out},{H}x{W})", diff)
        except Exception as e:
            record(f"conv2d_k1: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# TEST: Conv2d K=3 (shifted inputs approach)
# ════════════════════════════════════════════════════════════════════════
def test_conv2d_k3():
    print("\n── Conv2d K=3 (vae_conv2d_k3_shifted) ──────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_conv2d import vae_conv2d_k3_shifted
    except Exception as e:
        record("conv2d_k3: import", None, f"{type(e).__name__}: {e}")
        return

    # Production shapes from VAE decoder ResidualBlocks
    # (name, C_in, C_out, H, W)
    configs = [
        # Bottleneck ResBlocks
        ("resblock_1024→1024_30x52",   1024, 1024, 30, 52),
        # After first upsample
        ("resblock_1024→1024_60x104",  1024, 1024, 60, 104),
        # After second upsample — large spatial
        ("resblock_1024→512_120x208",  1024, 512,  120, 208),
        # Smallest channels, largest spatial (skip for now if too slow)
        # ("resblock_256→256_240x416",   256,  256,  240, 416),
        # Head conv
        ("head_conv_256→128_30x52",    256,  128,  30, 52),   # pad 128→128
    ]

    wrapped = wrap_nki(vae_conv2d_k3_shifted)

    for name, C_in, C_out, H, W in configs:
        try:
            torch.manual_seed(0)
            C_in_padded = ((C_in + 127) // 128) * 128
            C_out_padded = ((C_out + 127) // 128) * 128
            HW = H * W
            HW_padded = ((HW + 511) // 512) * 512

            # Create PyTorch Conv2d reference
            conv = nn.Conv2d(C_in, C_out, 3, padding=1, bias=True)
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)

            # Input
            x = torch.randn(1, C_in, H, W, dtype=torch.float32)

            # CPU reference
            with torch.no_grad():
                ref = conv(x)  # (1, C_out, H, W)
            ref_flat = ref.reshape(C_out, HW).to(torch.bfloat16)

            # Prepare NKI inputs
            x_flat = x.reshape(C_in, HW)

            # Build shifted inputs: (9*C_in, HW_padded)
            shifted = build_shifted_inputs(x_flat.to(torch.bfloat16), H, W)
            # Pad channels to 9*C_in_padded
            if C_in < C_in_padded:
                # Need to pad each of the 9 blocks
                chunks = shifted.reshape(9, C_in, -1)
                padded_chunks = F.pad(chunks, (0, 0, 0, C_in_padded - C_in))
                shifted = padded_chunks.reshape(9 * C_in_padded, -1)

            # Weight slices: (C_out, C_in*9) in blocked layout
            w_slices = build_weight_slices(conv.weight.data).to(torch.bfloat16)
            # Rearrange from (C_out, C_in*9) natural order to blocked: k_idx*C_in+c_in
            # build_weight_slices already produces this layout
            # Pad to (C_out_padded, C_in_padded*9)
            w_padded = torch.zeros(C_out_padded, C_in_padded * 9, dtype=torch.bfloat16)
            for k_idx in range(9):
                src_start = k_idx * C_in
                dst_start = k_idx * C_in_padded
                w_padded[:C_out, dst_start:dst_start + C_in] = w_slices[:, src_start:src_start + C_in]

            # Bias: (C_out_padded, 1)
            b = conv.bias.data.to(torch.bfloat16)
            b_padded = torch.zeros(C_out_padded, 1, dtype=torch.bfloat16)
            b_padded[:C_out, 0] = b

            # Run NKI kernel
            out = wrapped(
                shifted.to(DEVICE),
                w_padded.to(DEVICE),
                b_padded.to(DEVICE),
                HW  # num_positions
            ).cpu()

            # Trim to actual size
            out = out[:C_out, :HW]

            diff = (out.float() - ref_flat.float()).abs().max().item()
            record(f"conv2d_k3: {name} (Ci={C_in},Co={C_out},{H}x{W})", diff)
        except Exception as e:
            record(f"conv2d_k3: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# TEST: VAE Self-Attention
# ════════════════════════════════════════════════════════════════════════
def test_vae_attention():
    print("\n── VAE Self-Attention (vae_self_attention) ──────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from vae_attention import vae_self_attention
    except Exception as e:
        record("vae_attn: import", None, f"{type(e).__name__}: {e}")
        return

    def _sdpa_ref(q, k, v, scale):
        """CPU reference: q(1,d,Sq), k(1,d,Sk), v(1,Sk,d) → out(Sq,1,d)"""
        qa = q.permute(0, 2, 1).float()    # (1, Sq, d)
        ka = k.permute(0, 2, 1).float()    # (1, Sk, d)
        va = v.float()                      # (1, Sk, d)
        scores = torch.matmul(qa, ka.transpose(-1, -2)) * scale  # (1, Sq, Sk)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, va)        # (1, Sq, d)
        return out.permute(1, 0, 2).to(q.dtype)  # (Sq, 1, d)

    P = 128

    # Production shapes from VAE decoder AttentionBlock
    # (name, dim, H, W) — single head self-attention
    configs = [
        # Decoder middle block (bottleneck)
        ("dec_middle_d1024_30x52",   1024, 30, 52),
        # Encoder middle block (same architecture)
        ("enc_middle_d1024_30x52",   1024, 30, 52),
        # Smaller test case
        ("small_d256_16x16",          256, 16, 16),
        # Medium test
        ("med_d512_20x32",            512, 20, 32),
    ]

    wrapped = wrap_nki(vae_self_attention)

    for name, d, H, W in configs:
        try:
            torch.manual_seed(0)
            seq_raw = H * W
            # Pad seq to multiple of 128
            pad = (P - seq_raw % P) % P
            seq = seq_raw + pad

            scale = 1.0 / math.sqrt(d)

            q = torch.randn(1, d, seq_raw, dtype=torch.bfloat16)
            k = torch.randn(1, d, seq_raw, dtype=torch.bfloat16)
            v = torch.randn(1, seq_raw, d, dtype=torch.bfloat16)
            identity = torch.eye(128, dtype=torch.bfloat16)

            # CPU reference
            ref = _sdpa_ref(q, k, v, scale)  # (seq_raw, 1, d)

            # Pad for NKI kernel
            if pad > 0:
                q_padded = F.pad(q, (0, pad))
                k_padded = F.pad(k, (0, pad))
                v_padded = F.pad(v, (0, 0, 0, pad))
            else:
                q_padded = q
                k_padded = k
                v_padded = v

            out = wrapped(
                q_padded.to(DEVICE),
                k_padded.to(DEVICE),
                v_padded.to(DEVICE),
                identity.to(DEVICE),
                softmax_scale=scale
            ).cpu()

            # Trim to actual seq length
            out = out[:seq_raw]

            diff = (out.float() - ref.float()).abs().max().item()
            record(f"vae_attn: {name} (d={d},seq={seq_raw}→{seq})", diff)
        except Exception as e:
            record(f"vae_attn: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# TEST: Full ResidualBlock (NKI conv2d components vs PyTorch ResidualBlock)
# ════════════════════════════════════════════════════════════════════════
def test_residual_block_components():
    """Test that NKI conv2d_k3 + NKI conv2d_k1 can replicate a ResidualBlock.

    This is NOT a fused kernel test — it tests the building blocks that would
    be assembled in the modified vae2_2.py forward() to replace torch.compile.
    We test the individual conv components match PyTorch.
    """
    print("\n── ResidualBlock Component Test (conv2d_k3 + conv2d_k1) ────────")
    # This test just confirms the conv kernels are accurate enough
    # to build a ResidualBlock from. The actual wiring is in vae2_2.py.
    print("  (Covered by conv2d_k1 and conv2d_k3 tests above)")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("VAE NKI Kernel Test Suite")
    print(f"Device: {DEVICE}  Tolerance: max_diff < {TOL}")
    print("=" * 72)

    test_conv2d_k1()
    test_conv2d_k3()
    test_vae_attention()
    test_residual_block_components()

    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_total = len(results)
    all_ok = n_pass == n_total
    print(f"SUMMARY: {n_pass}/{n_total} passed  {'✅ ALL PASS' if all_ok else '❌ FAILURES'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)
