"""Wan2.2-TI2V-5B inference with TP8 on Neuron.

Uses torchrun --nproc_per_node=8 for tensor parallelism.
DiT is TP-sharded across 8 NeuronCores (2 NDs).
T5 on rank 4 (ND1:NC0), VAE on rank 0 (ND0:NC0).

Architecture (Wan2.2-TI2V-5B):
  dim=3072, 24 heads, 30 layers, ffn_dim=14336
  TP=8: 3 heads/rank, ~660M params/rank (~1.3GB bf16)
"""
import os
import sys
import time
import math
import random
import logging
import gc

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True,
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_mlir']:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Add Wan2.2 to path (pre-patched wan/ directory is in this repo)
WAN_DIR = os.environ.get("WAN_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAN_DIR)

# Add this repo's root to path for models.tp_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "8"))
T5_RANK = int(os.environ.get("T5_RANK", "4"))
VAE_RANK = 0

NEURON_DEVICE = torch.device("neuron")


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def main():
    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)

    if rank == 0:
        logger.info(f"Wan2.2-TI2V-5B TP={TP_DEGREE}, world_size={world_size}")

    # ── Import Wan modules ──
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.utils.utils import best_output_size, masks_like
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # Import TP utilities from rolling-forcing
    from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank

    config = WAN_CONFIGS['ti2v-5B']

    # ── Load T5 on T5_RANK — on Neuron + compiled (like rolling-forcing) ──
    if rank == T5_RANK:
        logger.info(f"Loading T5 on rank {T5_RANK} (on Neuron with torch.compile)...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
            tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
        # Move T5 model to Neuron and compile
        text_encoder.model = text_encoder.model.to(NEURON_DEVICE)
        text_encoder.model = torch.compile(text_encoder.model, backend='neuron', dynamic=False)
        logger.info(f"T5 loaded on Neuron with torch.compile (rank {T5_RANK})")
    else:
        text_encoder = None

    # ── Load VAE on VAE_RANK — on Neuron + compiled (like rolling-forcing) ──
    if rank == VAE_RANK:
        logger.info(f"Loading VAE on rank {VAE_RANK} (on Neuron with torch.compile)...")
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        # Move VAE model to Neuron and compile
        vae.model = vae.model.to(dtype=torch.bfloat16, device=NEURON_DEVICE)
        vae.model = torch.compile(vae.model, backend='neuron', dynamic=False)
        # Move VAE scale tensors to Neuron (used in encode/decode for normalization)
        vae.scale = [s.to(NEURON_DEVICE) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        logger.info(f"VAE loaded on Neuron with torch.compile (rank {VAE_RANK})")
    else:
        vae = None

    # ── Load DiT with TP sharding (all ranks) ──
    if rank == 0:
        logger.info(f"Loading WanModel from {MODEL_PATH}...")
    model = WanModel.from_pretrained(MODEL_PATH)
    model.eval().requires_grad_(False)

    # Initialize TP group and shard model
    init_tp_group(tp_degree=TP_DEGREE)
    tp_rank = get_tp_rank()
    shard_model_tp(model, tp_rank, TP_DEGREE)

    # Cast to bf16 then move to Neuron (Conv3d requires matching weight/bias/input dtypes)
    model = model.to(torch.bfloat16)
    model = model.to(NEURON_DEVICE)
    if rank == 0:
        logger.info("DiT moved to Neuron, compiling sub-modules...")

    # Compile pure sub-modules (skip time_embedding/time_projection — they receive fp32
    # sinusoidal input but have bf16 weights, which the Neuron compiler can't handle.
    # They're tiny modules so eager mode is fine.)
    model.patch_embedding = torch.compile(model.patch_embedding, backend='neuron', dynamic=False)
    model.text_embedding = torch.compile(model.text_embedding, backend='neuron', dynamic=False)
    # time_embedding and time_projection: keep in eager (fp32 input, bf16 weights)
    model.head = torch.compile(model.head, backend='neuron', dynamic=False)
    for block in model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"Compiled: patch_embed, text_embed, head, FFN x{len(model.blocks)} (time_embed/proj in eager)")

    dist.barrier()
    if rank == 0:
        logger.info("All models loaded!")

    # ── Encode prompt (T5 on T5_RANK, broadcast to all) ──
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, "
        "distant green hills, and a blue sky dotted with white clouds."
    )
    n_prompt = config.sample_neg_prompt

    # All ranks: allocate buffers on device (same operation everywhere)
    ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == T5_RANK:
        # T5 inference on Neuron — IDs go to device, model runs on device, output on device
        context = text_encoder([prompt], NEURON_DEVICE)
        context_null = text_encoder([n_prompt], NEURON_DEVICE)
        # Pad to fixed size [1, 512, 4096] for broadcast
        ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_tensor[0, :context[0].shape[0]] = context[0].to(torch.bfloat16)
        ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_null_tensor[0, :context_null[0].shape[0]] = context_null[0].to(torch.bfloat16)

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)
    context = [ctx_tensor]
    context_null = [ctx_null_tensor]

    # Free T5 after encoding to reclaim memory
    if rank == T5_RANK:
        del text_encoder
        gc.collect()

    if rank == 0:
        logger.info("Prompt encoded and broadcast to all ranks")

    # ── Prepare image and noise ──
    import torchvision.transforms.functional as TF

    image_path = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")
    img = Image.open(image_path).convert("RGB")

    vae_stride = config.vae_stride  # (4, 16, 16)
    patch_size = config.patch_size   # (1, 2, 2)
    frame_num = 17
    max_area = 704 * 1280

    ih, iw = img.height, img.width
    dh = patch_size[1] * vae_stride[1]
    dw = patch_size[2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(1)  # [C, 1, H, W]

    F = frame_num
    seq_len = (
        ((F - 1) // vae_stride[0] + 1)
        * (oh // vae_stride[1])
        * (ow // vae_stride[2])
        // (patch_size[1] * patch_size[2])
    )

    seed = 42
    seed_g = torch.Generator(device=torch.device("cpu"))
    seed_g.manual_seed(seed)

    # Wan2.2 VAE z_dim from config (48 for Wan2.2, 16 for Wan2.1)
    z_dim = config.vae_dim[0] if hasattr(config, 'vae_dim') else 48
    T_latent = (F - 1) // vae_stride[0] + 1  # 21 for 81 frames
    H_latent = oh // vae_stride[1]
    W_latent = ow // vae_stride[2]

    noise = torch.randn(
        z_dim, T_latent, H_latent, W_latent,
        dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

    # Encode image with VAE on rank 0, broadcast z to all ranks
    # VAE encode of a single image gives [z_dim, 1, H', W']
    z_img_shape = (z_dim, 1, H_latent, W_latent)

    # All ranks: allocate z buffer for image encoding
    z_img_device = torch.zeros(z_img_shape, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == VAE_RANK:
        # VAE encode on Neuron — input must be on device
        img_device = img_tensor.to(dtype=torch.bfloat16, device=NEURON_DEVICE)
        z = vae.encode([img_device])
        z_result = z[0].to(torch.bfloat16).contiguous()
        if z_result.device != NEURON_DEVICE:
            z_result = z_result.to(NEURON_DEVICE)
        z_img_device.copy_(z_result)

    dist.broadcast(z_img_device, src=VAE_RANK)

    # All ranks: move noise to neuron (same dtype everywhere)
    noise = noise.to(torch.bfloat16).to(NEURON_DEVICE)
    mask1, mask2 = masks_like([noise.cpu()], zero=True)
    mask2_device = [m.to(torch.bfloat16).to(NEURON_DEVICE) for m in mask2]

    # Build latent: image occupies temporal position 0, noise fills the rest
    # z_img_device is [z_dim, 1, H', W'], noise is [z_dim, T_latent, H', W']
    latent = noise.clone()
    latent = (1. - mask2_device[0]) * z_img_device.expand_as(noise) + mask2_device[0] * latent

    if rank == 0:
        logger.info(f"Image encoded, noise prepared. Starting denoising (50 steps)...")

    # ── Denoising loop ──
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(50, device=torch.device("cpu"), shift=config.sample_shift)
    timesteps = sample_scheduler.timesteps

    # Model expects context as list of 2D tensors [text_len, hidden_dim] (one per batch item)
    # Our tensors are [1, 512, 4096] — squeeze batch dim to get [512, 4096]
    arg_c = {'context': [context[0][0]], 'seq_len': seq_len}
    arg_null = {'context': [context_null[0][0]], 'seq_len': seq_len}

    start_time = time.time()
    for step_idx, t in enumerate(tqdm(timesteps, disable=(rank != 0))):
        latent_model_input = [latent]

        t_device = t.to(NEURON_DEVICE)
        temp_ts = (mask2_device[0][0][:, ::2, ::2] * t_device).flatten()
        temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_device])
        timestep = temp_ts.unsqueeze(0)

        noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]

        guide_scale = config.sample_guide_scale
        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step on CPU
        temp_x0 = sample_scheduler.step(
            noise_pred.cpu().unsqueeze(0), t,
            latent.cpu().unsqueeze(0),
            return_dict=False, generator=seed_g)[0]
        latent = temp_x0.squeeze(0).to(NEURON_DEVICE)
        latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent

    denoise_time = time.time() - start_time
    if rank == 0:
        logger.info(f"Denoising done in {denoise_time:.1f}s")

    # ── Decode with VAE on rank 0 ──
    if rank == VAE_RANK:
        logger.info("Decoding latents with VAE...")
        # Move VAE to CPU for decode (avoids Neuron dtype promotion issues:
        # z / scale promotes to float32 but Conv3d weights are bf16)
        vae.model = vae.model.cpu().float()
        vae.scale = [s.cpu().float() if isinstance(s, torch.Tensor) else s for s in vae.scale]
        x0 = [latent.cpu().float()]
        videos = vae.decode(x0)
        video = videos[0]

        # Save video
        from torchvision.io import write_video
        output_path = "/tmp/wan2_ti2v_output.mp4"
        video_cpu = video.cpu().float()
        video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
        if video_np.dim() == 4 and video_np.shape[0] == 3:
            video_np = video_np.permute(1, 2, 3, 0)  # [F, H, W, C]
        write_video(output_path, video_np, fps=24)
        logger.info(f"Video saved to {output_path}")

        # Copy to /var/mdl
        import shutil
        os.makedirs("/var/mdl/wan2_2_ti2v/outputs", exist_ok=True)
        shutil.copy(output_path, "/var/mdl/wan2_2_ti2v/outputs/")
        logger.info(f"Video copied to /var/mdl/wan2_2_ti2v/outputs/")

    dist.barrier()
    if rank == 0:
        logger.info(f"Total denoising time: {denoise_time:.1f}s")
        logger.info("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
