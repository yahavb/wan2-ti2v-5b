"""One-core parity check: const-max kernel vs exact attention, with this model's QK-norm gammas.

For each selected layer, builds q/k from RMS-normalized random activations times the real
norm_q/norm_k weights (rank 0's heads at TP_DEGREE, CFG batch folded in), runs
kernels/attention_const_max.py on the device, and compares against fp32 softmax attention
on CPU. Two sequence lengths: the run's (often not a multiple of 128) and the next
multiple of 128, so a partial-tile problem shows up as a gap between the two rows.

Usage (one NeuronCore): python3 scripts/prof/const_max_parity.py
Env: MODEL_PATH, TP_DEGREE, PARITY_SEQ (default 1080), PARITY_LAYERS, CONST_MAX_SOFTMAX_MAX.
"""
import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP = int(os.environ.get("TP_DEGREE", "8"))
SEQ = int(os.environ.get("PARITY_SEQ", "1080"))
LAYERS = [int(x) for x in os.environ.get("PARITY_LAYERS", "0,5,10,18,23,25,26,29").split(",")]
C = float(os.environ.get("CONST_MAX_SOFTMAX_MAX", "15.0"))
DIM, HEADS, D = 3072, 24, 128


def load_gammas():
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "diffusion_pytorch_model*.safetensors"))):
        with safe_open(f, "pt") as st:
            for n in st.keys():
                if ".self_attn.norm_q." in n or ".self_attn.norm_k." in n:
                    out[n] = st.get_tensor(n).float()
    return out


def rms_gamma(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g


def main():
    from kernels.attention_const_max import attention_const_max
    dev = torch.device("neuron")
    g = load_gammas()
    hpr = HEADS // TP
    print(f"const-max parity: TP={TP} heads/rank={hpr} batch=2 c={C} layers={LAYERS}")
    print(f"{'layer':>5} {'L':>6} {'rel_rms':>9} {'max_abs':>9} {'ref_rms':>9}  scaled-score max (sample)")
    torch.manual_seed(0)
    for b in LAYERS:
        gq = g[f"blocks.{b}.self_attn.norm_q.weight"]
        gk = g[f"blocks.{b}.self_attn.norm_k.weight"]
        for L in sorted({SEQ, -(-SEQ // 128) * 128}):
            xq, xk = torch.randn(2, L, DIM), torch.randn(2, L, DIM)
            q = rms_gamma(xq, gq)[..., :hpr * D].reshape(2, L, hpr, D)
            k = rms_gamma(xk, gk)[..., :hpr * D].reshape(2, L, hpr, D)
            v = torch.randn(2, L, hpr, D)
            # exact reference (fp32, CPU)
            qh, kh, vh = (t.permute(0, 2, 1, 3) for t in (q, k, v))
            scores = qh @ kh.transpose(-1, -2) / math.sqrt(D)
            ref = (scores.softmax(-1) @ vh).permute(0, 2, 1, 3)
            # kernel (bf16, device): [B*n, d, L] / [B*n, d, L] / [B*n, L, d]
            qn = q.permute(0, 2, 3, 1).reshape(2 * hpr, D, L).to(torch.bfloat16).to(dev)
            kn = k.permute(0, 2, 3, 1).reshape(2 * hpr, D, L).to(torch.bfloat16).to(dev)
            vn = v.permute(0, 2, 1, 3).reshape(2 * hpr, L, D).to(torch.bfloat16).to(dev)
            try:
                out = attention_const_max(qn, kn, vn, C, D ** -0.5)
                out = out.cpu().float().view(2, hpr, L, D).transpose(1, 2)
                err = out - ref
                rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
                print(f"{b:>5} {L:>6} {rel:>9.4f} {err.abs().max().item():>9.4f} "
                      f"{ref.pow(2).mean().sqrt().item():>9.4f}  {scores.max().item():.1f}", flush=True)
            except Exception as e:  # report and keep going
                print(f"{b:>5} {L:>6} KERNEL FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("const-max parity done (rel_rms ~0.01 = bf16-level; >0.05 = wrong)")


if __name__ == "__main__":
    main()
