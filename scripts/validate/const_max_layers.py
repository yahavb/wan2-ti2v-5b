"""Per layer and per TP4 rank shard: the const-max eligibility of models.fast_dit.const_max_layer_ok
(worst-case scaled score sqrt(d) * max|g_q * g_k| from the QK-norm weights) for the 30 DiT blocks.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 const_max_layers.py   Env: MODEL_PATH, TP (4)
"""
import glob
import os

import torch
from safetensors import safe_open

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP = int(os.environ.get("TP", "4"))
LIMIT = float(os.environ.get("CONST_MAX_HARD_LIMIT", "70"))
D = 128


def main():
    w = {}
    for f in sorted(glob.glob(os.path.join(MODEL_PATH, "diffusion_pytorch_model*.safetensors"))):
        with safe_open(f, "pt") as st:
            for n in st.keys():
                if ".self_attn.norm_q.weight" in n or ".self_attn.norm_k.weight" in n:
                    w[n] = st.get_tensor(n).float()
    ok_all = 0
    for i in range(30):
        gq = w[f"blocks.{i}.self_attn.norm_q.weight"].reshape(-1, D)
        gk = w[f"blocks.{i}.self_attn.norm_k.weight"].reshape(-1, D)
        per_head = (D ** 0.5) * (gq.abs().amax(1) * gk.abs().amax(1))
        shards = per_head.view(TP, -1).amax(1)
        ok = [bool(s <= LIMIT) for s in shards]
        ok_all += all(ok)
        print(f"block {i:2d}: hard bound per shard {[round(float(s), 1) for s in shards]} "
              f"-> {'all' if all(ok) else sum(ok)} of {TP} shards eligible", flush=True)
    print(f"{ok_all} of 30 blocks eligible on every shard (limit {LIMIT})")


if __name__ == "__main__":
    main()
