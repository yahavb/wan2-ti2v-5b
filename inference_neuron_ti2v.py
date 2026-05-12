"""Wan2.2-TI2V-5B inference with TP4 on Neuron.

Uses torchrun --nproc_per_node=4 for tensor parallelism.
DiT is TP-sharded across 4 NeuronCores.
T5 on rank 2 (ND1), VAE on rank 0 (ND0).

Architecture (Wan2.2-TI2V-5B):
  dim=3072, 24 heads, 30 layers, ffn_dim=14336
  TP=4: 6 heads/rank, ~1.25B params/rank (~2.5GB bf16)
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

# Add Wan2.2 to path
WAN_DIR = os.environ.get("WAN_DIR", "/tmp/Wan2.2")
sys.path.insert(0, WAN_DIR)

# Add this repo's root to path for models.tp_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "4"))
T5_RANK = int(os.environ.get("T5_RANK", "2"))
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
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.utils import best_output_size, masks_like
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # Import TP utilities from rolling-forcing
    from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank

    config = WAN_CONFIGS['ti2v-5B']

    # ── Load T5 on T5_RANK ──
    if rank == T5_RANK:
        logger.info(f"Loading T5 on rank {T5_RANK}...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
            tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
        logger.info("T5 loaded on CPU")
    else:
        text_encoder = None

    # ── Load VAE on VAE_RANK ──
    if rank == VAE_RANK:
        logger.info(f"Loading VAE on rank {VAE_RANK}...")
        vae = Wan2_1_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        logger.info("VAE loaded on CPU")
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

    # Move sharded model to Neuron
    model = model.to(NEURON_DEVICE)
    if rank == 0:
        logger.info("DiT moved to Neuron, compiling sub-modules...")

    # Compile pure sub-modules
    model.patch_embedding = torch.compile(model.patch_embedding, backend='neuron', dynamic=False)
    model.text_embedding = torch.compile(model.text_embedding, backend='neuron', dynamic=False)
    model.time_embedding = torch.compile(model.time_embedding, backend='neuron', dynamic=False)
    model.time_projection = torch.compile(model.time_projection, backend='neuron', dynamic=False)
    model.head = torch.compile(model.head, backend='neuron', dynamic=False)
    for block in model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"Compiled: patch_embed, text_embed, time_embed, time_proj, head, FFN x{len(model.blocks)}")

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

    if rank == T5_RANK:
        context = text_encoder([prompt], torch.device('cpu'))
        context_null = text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(NEURON_DEVICE) for t in context]
        context_null = [t.to(NEURON_DEVICE) for t in context_null]
        ctx_tensor = context[0].contiguous()
        ctx_null_tensor = context_null[0].contiguous()
    else:
        # Allocate buffers - T5 output is [1, 512, 4096]
        ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)
    context = [ctx_tensor]
    context_null = [ctx_null_tensor]

    if rank == 0:
        logger.info("Prompt encoded and broadcast to all ranks")

    # ── Prepare image and noise ──
    import torchvision.transforms.functional as TF

    image_path = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")
    img = Image.open(image_path).convert("RGB")

    vae_stride = config.vae_stride  # (4, 16, 16)
    patch_size = config.patch_size   # (1, 2, 2)
    frame_num = 81
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

    z_dim = 16  # Wan VAE z_dim
    noise = torch.randn(
        z_dim, (F - 1) // vae_stride[0] + 1,
        oh // vae_stride[1], ow // vae_stride[2],
        dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

    # Encode image with VAE on rank 0
    if rank == VAE_RANK:
        z = vae.encode([img_tensor])
        z_tensor = z[0].contiguous()
    else:
        z_shape = (z_dim, (F - 1) // vae_stride[0] + 1, oh // vae_stride[1], ow // vae_stride[2])
        z_tensor = torch.zeros(z_shape, dtype=torch.float32)

    # Broadcast z to all ranks
    z_tensor_device = z_tensor.to(NEURON_DEVICE)
    dist.broadcast(z_tensor_device, src=VAE_RANK)

    # Move noise to neuron
    noise = noise.to(NEURON_DEVICE)
    mask1, mask2 = masks_like([noise.cpu()], zero=True)
    mask2_device = [m.to(NEURON_DEVICE) for m in mask2]

    latent = noise
    latent = (1. - mask2_device[0]) * z_tensor_device + mask2_device[0] * latent

    if rank == 0:
        logger.info(f"Image encoded, noise prepared. Starting denoising (50 steps)...")

    # ── Denoising loop ──
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(50, device=torch.device("cpu"), shift=config.sample_shift)
    timesteps = sample_scheduler.timesteps

    arg_c = {'context': [context[0]], 'seq_len': seq_len}
    arg_null = {'context': context_null, 'seq_len': seq_len}

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
        latent = (1. - mask2_device[0]) * z_tensor_device + mask2_device[0] * latent

    denoise_time = time.time() - start_time
    if rank == 0:
        logger.info(f"Denoising done in {denoise_time:.1f}s")

    # ── Decode with VAE on rank 0 ──
    if rank == VAE_RANK:
        logger.info("Decoding latents with VAE...")
        x0 = [latent.cpu()]
        videos = vae.decode(x0)
        video = videos[0]

        # Save video
        from torchvision.io import write_video
        output_path = "/tmp/wan2_ti2v_output.mp4"
        video_np = ((video.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
        if video_np.dim() == 4 and video_np.shape[0] == 3:
            video_np = video_np.permute(1, 2, 3, 0)  # [F, H, W, C]
        write_video(output_path, video_np.cpu(), fps=24)
        logger.info(f"Video saved to {output_path}")

        # Copy to S3 PVC
        import shutil
        os.makedirs("/var/mdl/wan2_2_ti2v/outputs", exist_ok=True)
        shutil.copy(output_path, "/var/mdl/wan2_2_ti2v/outputs/")
        logger.info("Video copied to S3 PVC")

    dist.barrier()
    if rank == 0:
        logger.info(f"Total denoising time: {denoise_time:.1f}s")
        logger.info("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
