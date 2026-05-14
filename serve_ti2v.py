"""Wan2.2-TI2V-5B FastAPI server with TP-8 on Neuron.

Uses torchrun --nproc_per_node=8 for tensor parallelism.
DiT is TP-sharded across 8 NeuronCores.
T5 on rank T5_RANK, VAE on rank 0.

API:
  POST /generate  — {"prompt": "...", "image_url": "...", "num_steps": 10, "seed": 42}
                    Returns {"video": "<base64 mp4>", "execution_time": 123.4, "frames": 81}
  GET  /health    — {"status": "ok"}
  GET  /readiness — {"status": "ready"} (after warmup)
"""
import os
import sys
import time
import math
import random
import logging
import gc
import base64
import threading
import tempfile

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
WAN_DIR = os.environ.get("WAN_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAN_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "8"))
T5_RANK = int(os.environ.get("T5_RANK", "2"))
VAE_RANK = 0
VAE_TP_DEGREE = int(os.environ.get("VAE_TP_DEGREE", "1"))
VAE_TP_RANKS = list(range(VAE_TP_DEGREE))

NEURON_DEVICE = torch.device("neuron")

# Shared state for inter-rank communication
INPUTS_PATH = "/tmp/current_serve_inputs.pt"
RESULT_PATH = "/tmp/current_serve_result.mp4"
_model_ready = False


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def load_models(rank, world_size):
    """Load T5, VAE, DiT with TP sharding — same as inference_neuron_ti2v.py."""
    global _model_ready

    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
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
        text_encoder.model = text_encoder.model.to(NEURON_DEVICE)
        logger.info(f"T5 loaded on Neuron (rank {T5_RANK})")
    else:
        text_encoder = None

    # ── Load VAE ──
    if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
        from models.vae_tp import create_vae_tp_group, shard_vae_model_tp
        if rank == VAE_TP_RANKS[0]:
            logger.info(f"Loading VAE with TP={VAE_TP_DEGREE} on ranks {VAE_TP_RANKS}...")
        os.environ["USE_NKI_VAE"] = "0"
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        create_vae_tp_group(VAE_TP_RANKS)
        vae_tp_rank = VAE_TP_RANKS.index(rank)
        shard_vae_model_tp(vae.model, vae_tp_rank, VAE_TP_DEGREE)
        logger.info(f"VAE TP={VAE_TP_DEGREE} sharded on rank {rank}")
    elif VAE_TP_DEGREE <= 1 and rank == VAE_RANK:
        logger.info(f"Loading VAE on rank {VAE_RANK}...")
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        logger.info("VAE loaded on Neuron")
    else:
        vae = None

    # ── Load DiT with TP sharding (all ranks) ──
    if rank == 0:
        logger.info(f"Loading WanModel from {MODEL_PATH}...")
    model = WanModel.from_pretrained(MODEL_PATH)
    model.eval().requires_grad_(False)

    init_tp_group(tp_degree=TP_DEGREE)
    tp_rank = get_tp_rank()
    shard_model_tp(model, tp_rank, TP_DEGREE)

    model = model.to(torch.bfloat16)
    model = model.to(NEURON_DEVICE)
    if rank == 0:
        logger.info("DiT moved to Neuron, compiling sub-modules...")

    model.patch_embedding = torch.compile(model.patch_embedding, backend='neuron', dynamic=False)
    model.text_embedding = torch.compile(model.text_embedding, backend='neuron', dynamic=False)
    model.head = torch.compile(model.head, backend='neuron', dynamic=False)
    for block in model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"Compiled: patch_embed, text_embed, head, FFN x{len(model.blocks)}")

    dist.barrier()
    if rank == 0:
        logger.info("All models loaded!")

    return config, text_encoder, vae, model


def encode_prompt(rank, text_encoder, config, prompt, n_prompt):
    """Encode prompt with T5 on T5_RANK, broadcast to all ranks."""
    ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == T5_RANK:
        context = text_encoder([prompt], NEURON_DEVICE)
        context_null = text_encoder([n_prompt], NEURON_DEVICE)
        ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_tensor[0, :context[0].shape[0]] = context[0].to(torch.bfloat16)
        ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_null_tensor[0, :context_null[0].shape[0]] = context_null[0].to(torch.bfloat16)

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)
    return [ctx_tensor], [ctx_null_tensor]


def encode_image(rank, vae, img_tensor, z_dim, H_latent, W_latent):
    """Encode image with VAE on rank 0, broadcast to all."""
    z_img_shape = (z_dim, 1, H_latent, W_latent)
    z_img_device = torch.zeros(z_img_shape, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == VAE_RANK:
        img_neuron = img_tensor.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        z = vae.encode([img_neuron])
        z_result = z[0].to(torch.bfloat16).contiguous()
        if z_result.device != NEURON_DEVICE:
            z_result = z_result.to(NEURON_DEVICE)
        z_img_device.copy_(z_result)

    dist.broadcast(z_img_device, src=VAE_RANK)
    return z_img_device


def run_denoising(rank, model, config, latent, mask2_device, z_img_device,
                  context, context_null, seq_len, num_steps, seed):
    """Run the denoising loop — all ranks participate."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    seed_g = torch.Generator(device=torch.device("cpu"))
    seed_g.manual_seed(seed)

    arg_c = {'context': [context[0][0]], 'seq_len': seq_len}
    arg_null = {'context': [context_null[0][0]], 'seq_len': seq_len}

    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(num_steps, device=torch.device("cpu"), shift=config.sample_shift)
    timesteps = sample_scheduler.timesteps

    for step_idx, t in enumerate(tqdm(timesteps, disable=(rank != 0), desc="Denoising")):
        latent_model_input = [latent]
        t_device = t.to(NEURON_DEVICE)
        temp_ts = (mask2_device[0][0][:, ::2, ::2] * t_device).flatten()
        temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_device])
        timestep = temp_ts.unsqueeze(0)

        noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]

        guide_scale = config.sample_guide_scale
        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

        temp_x0 = sample_scheduler.step(
            noise_pred.cpu().unsqueeze(0), t,
            latent.cpu().unsqueeze(0),
            return_dict=False, generator=seed_g)[0]
        latent = temp_x0.squeeze(0).to(NEURON_DEVICE)
        latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent

    return latent


def decode_and_save_video(rank, vae, latent, output_path):
    """Decode latent with VAE and save as mp4."""
    import imageio

    if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
        x0 = [latent.to(torch.bfloat16)]
        videos = vae.decode(x0)
        if rank == VAE_RANK:
            video = videos[0]
            video_cpu = video.cpu().float()
            video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
            if video_np.dim() == 4 and video_np.shape[0] == 3:
                video_np = video_np.permute(1, 2, 3, 0)
            frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
            imageio.mimwrite(output_path, frames, fps=24, codec='libx264')
            return len(frames)
    elif rank == VAE_RANK:
        x0 = [latent.to(torch.bfloat16)]
        videos = vae.decode(x0)
        video = videos[0]
        video_cpu = video.cpu().float()
        video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
        if video_np.dim() == 4 and video_np.shape[0] == 3:
            video_np = video_np.permute(1, 2, 3, 0)
        frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
        imageio.mimwrite(output_path, frames, fps=24, codec='libx264')
        return len(frames)
    return 0


def prepare_latent(config, img_tensor, frame_num, oh, ow, seed):
    """Prepare noise and latent from image tensor."""
    from wan.utils.utils import masks_like

    vae_stride = config.vae_stride
    patch_size = config.patch_size

    z_dim = config.vae_dim[0] if hasattr(config, 'vae_dim') else 48
    T_latent = (frame_num - 1) // vae_stride[0] + 1
    H_latent = oh // vae_stride[1]
    W_latent = ow // vae_stride[2]

    seq_len = (
        T_latent
        * (oh // vae_stride[1])
        * (ow // vae_stride[2])
        // (patch_size[1] * patch_size[2])
    )

    seed_g = torch.Generator(device=torch.device("cpu"))
    seed_g.manual_seed(seed)

    noise = torch.randn(
        z_dim, T_latent, H_latent, W_latent,
        dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

    return noise, z_dim, T_latent, H_latent, W_latent, seq_len


def run_full_pipeline(rank, config, text_encoder, vae, model,
                      prompt, image_url_or_path, num_steps=10, seed=42, frame_num=81):
    """Full TI2V pipeline: encode prompt → encode image → denoise → decode video."""
    import torchvision.transforms.functional as TF
    from wan.utils.utils import best_output_size, masks_like
    import urllib.request

    total_start = time.time()

    # ── Load image ──
    if rank == 0:
        logger.info(f"Loading image: {image_url_or_path}")

    if image_url_or_path.startswith("http"):
        with urllib.request.urlopen(image_url_or_path) as response:
            image_data = response.read()
        img = Image.open(torch.io.BytesIO(image_data) if hasattr(torch.io, 'BytesIO') else __import__('io').BytesIO(image_data)).convert("RGB")
    else:
        img = Image.open(image_url_or_path).convert("RGB")

    vae_stride = config.vae_stride
    patch_size = config.patch_size
    max_area = 480 * 832

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

    # ── Prepare latent ──
    noise, z_dim, T_latent, H_latent, W_latent, seq_len = prepare_latent(
        config, img_tensor, frame_num, oh, ow, seed)

    # ── Encode image with VAE ──
    z_img_device = encode_image(rank, vae, img_tensor, z_dim, H_latent, W_latent)

    # ── Encode prompt with T5 ──
    n_prompt = config.sample_neg_prompt
    context, context_null = encode_prompt(rank, text_encoder, config, prompt, n_prompt)

    # ── Prepare noise and masks ──
    noise = noise.to(torch.bfloat16).to(NEURON_DEVICE)
    mask1, mask2 = masks_like([noise.cpu()], zero=True)
    mask2_device = [m.to(torch.bfloat16).to(NEURON_DEVICE) for m in mask2]

    latent = noise.clone()
    latent = (1. - mask2_device[0]) * z_img_device.expand_as(noise) + mask2_device[0] * latent

    # ── Denoise ──
    denoise_start = time.time()
    latent = run_denoising(rank, model, config, latent, mask2_device, z_img_device,
                           context, context_null, seq_len, num_steps, seed)
    denoise_time = time.time() - denoise_start

    # ── Decode with VAE ──
    vae_start = time.time()
    output_path = RESULT_PATH
    num_frames = decode_and_save_video(rank, vae, latent, output_path)
    vae_time = time.time() - vae_start

    dist.barrier()
    total_time = time.time() - total_start

    if rank == 0:
        logger.info(f"Pipeline done: denoise={denoise_time:.1f}s, vae={vae_time:.1f}s, total={total_time:.1f}s")

    return total_time, num_frames


def main():
    global _model_ready

    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)

    if rank == 0:
        logger.info(f"Wan2.2-TI2V-5B Server TP={TP_DEGREE}, world_size={world_size}")

    # ── Load all models ──
    config, text_encoder, vae, model = load_models(rank, world_size)

    # ── Warmup: run one inference to trigger compilation ──
    if rank == 0:
        logger.info("=" * 60)
        logger.info("  WARMUP: Running initial inference (triggers compilation)")
        logger.info("=" * 60)

    warmup_prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression."
    )
    warmup_image = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")

    warmup_time, _ = run_full_pipeline(
        rank, config, text_encoder, vae, model,
        prompt=warmup_prompt,
        image_url_or_path=warmup_image,
        num_steps=int(os.environ.get("NUM_STEPS", "10")),
        seed=42,
        frame_num=81,
    )

    if rank == 0:
        logger.info(f"Warmup complete in {warmup_time:.1f}s")

    _model_ready = True
    dist.barrier()

    if rank == 0:
        logger.info("[READY] Model warmed up, starting HTTP server...")

    # ═══════════════════════════════════════════════════════════════
    # SERVING: FastAPI on rank 0, all ranks participate in generate
    # ═══════════════════════════════════════════════════════════════

    if rank == 0:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import Optional

        app = FastAPI(title="Wan2.2-TI2V-5B (PyTorch Native, TP-8)")
        inference_lock = threading.Lock()

        class GenerateRequest(BaseModel):
            prompt: str
            image_url: Optional[str] = None
            image_base64: Optional[str] = None

        class GenerateResponse(BaseModel):
            video: str  # base64 encoded mp4
            execution_time: float
            frames: int

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/readiness")
        def readiness():
            if not _model_ready:
                raise HTTPException(status_code=503, detail="Model not ready")
            return {"status": "ready"}

        @app.post("/generate")
        def generate(request: GenerateRequest):
            if not _model_ready:
                raise HTTPException(status_code=503, detail="Model not ready")

            if not request.image_url and not request.image_base64:
                raise HTTPException(status_code=400, detail="Either image_url or image_base64 is required")

            with inference_lock:
                # Resolve image source
                if request.image_base64:
                    # Save base64 image to temp file
                    import io as _io
                    img_bytes = base64.b64decode(request.image_base64)
                    img_path = "/tmp/serve_input_image.png"
                    Image.open(_io.BytesIO(img_bytes)).convert("RGB").save(img_path)
                    image_source = img_path
                    logger.info(f"[REQUEST] prompt='{request.prompt[:80]}...', image=base64 ({len(request.image_base64)} chars)")
                else:
                    image_source = request.image_url
                    logger.info(f"[REQUEST] prompt='{request.prompt[:80]}...', image_url='{request.image_url[:80]}'")

                # Hardcoded params — must match warmup to avoid recompilation
                req_data = {
                    'prompt': request.prompt,
                    'image_url': image_source,
                    'num_steps': 10,
                    'seed': 42,
                    'frame_num': 81,
                }
                torch.save(req_data, INPUTS_PATH)

                # Broadcast signal to other ranks: [num_steps, seed, frame_num, 1=generate]
                signal = torch.tensor(
                    [req_data['num_steps'], req_data['seed'], req_data['frame_num'], 1],
                    dtype=torch.long).to(NEURON_DEVICE)
                dist.broadcast(signal, src=0)

                # Run full pipeline (all ranks participate)
                exec_time, num_frames = run_full_pipeline(
                    rank, config, text_encoder, vae, model,
                    prompt=req_data['prompt'],
                    image_url_or_path=req_data['image_url'],
                    num_steps=req_data['num_steps'],
                    seed=req_data['seed'],
                    frame_num=req_data['frame_num'],
                )

                # Read output video and encode as base64
                with open(RESULT_PATH, 'rb') as f:
                    video_bytes = f.read()
                video_b64 = base64.b64encode(video_bytes).decode('utf-8')

                logger.info(f"[RESPONSE] {exec_time:.1f}s, {num_frames} frames, {len(video_bytes)} bytes")

                return GenerateResponse(
                    video=video_b64,
                    execution_time=round(exec_time, 2),
                    frames=num_frames,
                )

        def start_server():
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        logger.info(f"\n[SERVER] FastAPI running on port 8000")

        # Rank 0 main thread: keep alive
        while True:
            time.sleep(0.1)

    else:
        # Non-rank-0: wait for broadcast signals from rank 0
        while True:
            try:
                signal = torch.tensor([10, 42, 81, 0], dtype=torch.long).to(NEURON_DEVICE)
                dist.broadcast(signal, src=0)

                num_steps = signal[0].item()
                seed = signal[1].item()
                frame_num = signal[2].item()
                action = signal[3].item()

                if action == 1:
                    # Load request data
                    req_data = torch.load(INPUTS_PATH, weights_only=False)
                    run_full_pipeline(
                        rank, config, text_encoder, vae, model,
                        prompt=req_data['prompt'],
                        image_url_or_path=req_data['image_url'],
                        num_steps=req_data['num_steps'],
                        seed=req_data['seed'],
                        frame_num=req_data['frame_num'],
                    )
            except Exception as e:
                logger.error(f"[Rank {rank}] Error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    main()
