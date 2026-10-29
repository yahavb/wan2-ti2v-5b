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
SP_DEGREE = int(os.environ.get("SP_DEGREE", "1"))
T5_RANK = int(os.environ.get("T5_RANK", "8"))
VAE_RANK = 0
PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, "
    "distant green hills, and a blue sky dotted with white clouds."
)
VAE_TP_DEGREE = int(os.environ.get("VAE_TP_DEGREE", "1"))  # 1=single rank (existing), 2=channel-parallel TP
VAE_TP_RANKS = list(range(VAE_TP_DEGREE))  # e.g. [0] for TP=1, [0,1] for TP=2
# VAE_TILED=1: every rank loads the VAE and decodes a share of overlapping spatial tiles
# (models/vae_tiled.py); rank 0 blends. VAE_TILE_OVERLAP is in latent units.
VAE_TILED = os.environ.get("VAE_TILED", "0") == "1"
VAE_TILE_OVERLAP = int(os.environ.get("VAE_TILE_OVERLAP", "8"))
# VAE_STRIPS=1 (with VAE_TILED + VAE_STREAM): exact 8x1 horizontal strips instead of overlapping
# tiles; every spatial conv exchanges one row with each neighbour (models/vae_tiled.py), so the
# frames match an untiled decode up to rounding and no rank decodes another's rows.
VAE_STRIPS = os.environ.get("VAE_STRIPS", "0") == "1"
# VAE_STREAM=1 (with VAE_TILED): ship each decoded latent-frame chunk to rank 0 right away so the
# blend and the mp4 encode run while the tiles are still decoding. Same frames as VAE_STREAM=0.
VAE_STREAM = os.environ.get("VAE_STREAM", "0") == "1"
# x264 settings for the output mp4. imageio's default is preset medium, crf 25 (~1.1s for
# 81 frames at 704x544); veryfast + crf 20 is ~8x faster at about the same PSNR.
MP4_PRESET = os.environ.get("MP4_PRESET", "veryfast")
MP4_CRF = os.environ.get("MP4_CRF", "20")
MP4_KW = dict(quality=None, ffmpeg_params=["-preset", MP4_PRESET, "-crf", MP4_CRF])
USE_NST_SELF_ATTN = os.environ.get("USE_NST_SELF_ATTN", "0") == "1"
# DIT_FAST=1: block-compiled DiT with in-graph collectives + batched CFG (models/fast_dit.py).
# DIT_FAST=0: original eager path (per-FFN NEFFs, eager all-reduces, 2 forwards/step).
DIT_FAST = os.environ.get("DIT_FAST", "1") == "1"
# CFG_PARALLEL=1 (DIT_FAST only): split the ranks into two TP groups of world/2; group 0 runs the
# cond branch, group 1 the uncond branch, each at batch 1. Halves the bytes of every TP
# all-reduce and shrinks the group; the two predictions are combined in the head.
CFG_PARALLEL = os.environ.get("CFG_PARALLEL", "0") == "1"
CFG_GROUPS = os.environ.get("CFG_GROUPS", "contiguous")  # or "interleave" (see init below)

NEURON_DEVICE = torch.device("neuron")
# DUMP_DIR: rank 0 saves the last run's final latent (latent.pt) and uint8 frames (frames.npy)
# there, for comparing configurations numerically. Not timed (outside the convert timer).
DUMP_DIR = os.environ.get("DUMP_DIR", "")
# RUN_HASH=1 (determinism probe): after each run's timers, every rank logs an md5 of its final latent
# and rank 0 of the mp4; after the last run, md5s of the encoder outputs, every block's cross-attn
# K/V and per-parameter weight sums. All copies run outside the timed region.
# RUN_HASH_KEEP=<prefix>: also keep each run's mp4 as <prefix>_run<i>.mp4.
RUN_HASH = os.environ.get("RUN_HASH", "0") == "1"
# MEM_LOG=1: every rank prints its device memory (torch allocator) before and after each VAE decode.
MEM_LOG = os.environ.get("MEM_LOG", "0") == "1"
# STEP_TIMING=1: rank 0 logs, per denoising step, how long the host waited for the DiT and how
# long the host-side scheduler round trip took (no extra device syncs).
STEP_TIMING = os.environ.get("STEP_TIMING", "0") == "1"
# MP4_VFORK=1: start the mp4 encoder (ffmpeg) without imageio's preexec_fn, so Python spawns it with
# vfork instead of fork. Forking this process (tens of GB resident) blocks the other threads while
# the page tables are copied; the runtime's completion thread stalled ~90 ms once per decode.
if os.environ.get("MP4_VFORK", "0") == "1":
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
# DEVICE_SCHEDULER=1 (DIT_FAST): the UniPC step runs on the device tensors (models/device_unipc.py);
# its scalar coefficients stay on the CPU. No per-step device->host->device round trip, so the host
# queues the next step while the device is still busy. Same math; device elementwise rounding.
DEVICE_SCHEDULER = os.environ.get("DEVICE_SCHEDULER", "0") == "1"


def _dump_outputs(out_dir, latent, video_np):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(latent.detach().float().cpu(), os.path.join(out_dir, "latent.pt"))
    np.save(os.path.join(out_dir, "frames.npy"), video_np)


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    print(f"[mem] rank {dist.get_rank()} local_rank {local_rank} "
          f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')}", flush=True)
    return dist.get_rank(), dist.get_world_size()


def device_mem_str():
    """Allocated / reserved device memory of this rank, in GB (best effort)."""
    for mod in (getattr(torch, "neuron", None), getattr(torch, "accelerator", None)):
        try:
            return (f"alloc={mod.memory_allocated() / 2**30:.2f}GB "
                    f"reserved={mod.memory_reserved() / 2**30:.2f}GB "
                    f"max_alloc={mod.max_memory_allocated() / 2**30:.2f}GB")
        except Exception:
            continue
    return "n/a"


FRAME_NUM = int(os.environ.get("FRAME_NUM", "81"))
# Output size; unset = pick from the input image aspect within 480*832 area.
OUT_HEIGHT = int(os.environ.get("HEIGHT", "0"))
OUT_WIDTH = int(os.environ.get("WIDTH", "0"))
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "unknown")
LNC_CONFIG = os.environ.get("LNC_CONFIG", "unknown")
NUM_NEURON_DEVICES = os.environ.get("NUM_NEURON_DEVICES", "unknown")
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "0")
USE_NKI_VAE = os.environ.get("USE_NKI_VAE", "0")


def main():
    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)
    if rank == 0:
        # torchrun sets OMP_NUM_THREADS=1; rank 0 sums the VAE tiles and converts the video on CPU.
        torch.set_num_threads(int(os.environ.get("RANK0_CPU_THREADS", "16")))


    global T5_RANK
    if T5_RANK >= world_size:
        T5_RANK = world_size - 1
    if DIT_FAST:
        assert SP_DEGREE == 1, "DIT_FAST requires SP_DEGREE=1"
    # CPU group for gathering decoded VAE tiles (small, once per video).
    cpu_group = dist.new_group(backend="gloo") if VAE_TILED else None

    # Initialize SP/TP parallel groups if SP > 1
    if SP_DEGREE > 1:
        from models.parallel_state import init_parallel_groups
        init_parallel_groups(sp_degree=SP_DEGREE, tp_degree=TP_DEGREE)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("  Wan2.2-TI2V-5B  BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"  Instance:        {INSTANCE_TYPE}")
        logger.info(f"  LNC config:      {LNC_CONFIG}")
        logger.info(f"  NeuronDevices:   {NUM_NEURON_DEVICES}")
        logger.info(f"  TP degree:       {TP_DEGREE}")
        logger.info(f"  SP degree:       {SP_DEGREE}")
        logger.info(f"  World size:      {world_size}")
        logger.info(f"  NKI kernels:     DiT={USE_NKI_KERNELS}, VAE={USE_NKI_VAE}")
        logger.info(f"  NST self-attn:   {USE_NST_SELF_ATTN}")
        logger.info(f"  VAE TP degree:   {VAE_TP_DEGREE}")
        logger.info(f"  T5 rank:         {T5_RANK}")
        logger.info(f"  Frame count:     {FRAME_NUM}")
        logger.info("=" * 70)

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
        logger.info(f"T5 loaded on Neuron eager (rank {T5_RANK})")
        # Encode both prompts now and free T5 before the VAE and DiT weights land on this rank.
        # Freed after them, T5's ~11 GB left ~1.9 GB of cached fragments here for the whole run,
        # which the HBM-stack neighbour then ran short of during the VAE.
        t5_out = [text_encoder([p], NEURON_DEVICE)[0].to(torch.bfloat16).cpu()
                  for p in (PROMPT, config.sample_neg_prompt)]
        del text_encoder
        gc.collect()
        torch.neuron.empty_cache()
    text_encoder = None

    # ── Load VAE ──
    # VAE_TP_DEGREE=1 (default): single rank, NKI kernels, existing path
    # VAE_TP_DEGREE=2: 2-way channel TP, no NKI, PyTorch SDPA fallback
    if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
        # VAE TP path: load on all VAE TP ranks, shard decoder
        from models.vae_tp import create_vae_tp_group, shard_vae_model_tp
        if rank == VAE_TP_RANKS[0]:
            logger.info(f"Loading VAE with TP={VAE_TP_DEGREE} on ranks {VAE_TP_RANKS}...")
        # Disable NKI for TP (channel dims don't align to P=128 tile size)
        os.environ["USE_NKI_VAE"] = "0"
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        # Create VAE TP group and shard decoder
        create_vae_tp_group(VAE_TP_RANKS)
        vae_tp_rank = VAE_TP_RANKS.index(rank)
        shard_vae_model_tp(vae.model, vae_tp_rank, VAE_TP_DEGREE)
        logger.info(f"VAE TP={VAE_TP_DEGREE} sharded on rank {rank} (vae_tp_rank={vae_tp_rank})")
    elif VAE_TP_DEGREE <= 1 and (rank == VAE_RANK or VAE_TILED):
        # Original single-rank path with NKI kernels
        logger.info(f"Loading VAE on rank {VAE_RANK} (on Neuron with NKI kernels)...")
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        # Pack the nkilib conv filters now, so the first decode runs the same kernels (and
        # VAE_FUSE_RES rounding) as every later one (see Decoder3d.prepack_nki_filters).
        n_packed = vae.prepack_nki_filters()
        logger.info(f"VAE loaded on Neuron with NKI kernels ({n_packed} decoder convs pre-packed)")
    else:
        vae = None

    # ── Load DiT with TP sharding (all ranks) ──
    if rank == 0:
        logger.info(f"Loading WanModel from {MODEL_PATH}...")
    model = WanModel.from_pretrained(MODEL_PATH)
    model.eval().requires_grad_(False)

    # Initialize TP group and shard model
    dit_tp = TP_DEGREE // 2 if CFG_PARALLEL else TP_DEGREE
    # CFG_GROUPS=interleave: CFG-parallel TP groups on alternating ranks ([0,2,4,6] cond,
    # [1,3,5,7] uncond), so each group has one core on every HBM stack (block 19.8 vs 20.6 ms).
    interleave = CFG_PARALLEL and CFG_GROUPS == "interleave"
    tp_groups = ([list(range(0, world_size, 2)), list(range(1, world_size, 2))] if interleave else None)
    init_tp_group(tp_degree=dit_tp, sp_degree=SP_DEGREE, groups=tp_groups)
    tp_rank = get_tp_rank()
    # FAST_LOCAL_CROSS=1 (with CFG_PARALLEL + FAST_SP): keep an unsharded copy of every block's
    # cross-attention so it runs on the token shard with all heads, without collectives.
    full_cross = None
    if DIT_FAST and CFG_PARALLEL and os.environ.get("FAST_SP", "0") == "1" \
            and os.environ.get("FAST_LOCAL_CROSS", "0") == "1":
        import copy
        full_cross = [copy.deepcopy(b.cross_attn).to(torch.bfloat16) for b in model.blocks]
    # FAST_A2A_O=1 (on top of FAST_LOCAL_CROSS): keep an unsharded copy of every block's
    # self-attention O-projection; the attention output reaches it by all-to-all (models/fast_dit.py).
    full_o = None
    if full_cross is not None and os.environ.get("FAST_A2A_O", "0") == "1":
        full_o = [copy.deepcopy(b.self_attn.o).to(torch.bfloat16) for b in model.blocks]
    shard_model_tp(model, tp_rank, dit_tp)
    cfg_group, pair_group = None, None
    if CFG_PARALLEL:
        assert DIT_FAST and SP_DEGREE == 1 and world_size == TP_DEGREE and TP_DEGREE % 2 == 0, \
            "CFG_PARALLEL needs DIT_FAST=1, SP_DEGREE=1 and an even world size"
        # Pair = the two ranks with the same TP rank (same weight shard and token shard), one
        # per CFG group; the head sums the weighted cond/uncond predictions over it.
        if interleave:
            cfg_group = rank % 2
            pairs = [[2 * i, 2 * i + 1] for i in range(dit_tp)]
        else:
            cfg_group = rank // dit_tp
            pairs = [[i, i + dit_tp] for i in range(dit_tp)]
        for p in pairs:  # every rank creates every pair group, same order
            g = dist.new_group(p)
            if rank in p:
                pair_group = g
        if rank == 0:
            logger.info(f"CFG_PARALLEL=1: 2 TP groups of {dit_tp}, layout {CFG_GROUPS} "
                        f"(cond {tp_groups[0] if interleave else list(range(dit_tp))})")

    # Patch self-attention for SP if enabled
    if SP_DEGREE > 1:
        from models.sp_patch import patch_model_for_sp
        patch_model_for_sp(model)

    # Cast to bf16 then move to Neuron (Conv3d requires matching weight/bias/input dtypes)
    if full_cross is not None:
        # The sharded cross-attention is unused in this mode; drop its weights before the move.
        for b in model.blocks:
            for lin_ in (b.cross_attn.q, b.cross_attn.k, b.cross_attn.v, b.cross_attn.o):
                lin_.weight = torch.nn.Parameter(lin_.weight.new_empty(0), requires_grad=False)
    model = model.to(torch.bfloat16)
    model = model.to(NEURON_DEVICE)
    if rank == 0:
        logger.info("DiT moved to Neuron, compiling sub-modules...")

    if DIT_FAST:
        # Compilation happens in FastWanDiT once the latent grid is known.
        if rank == 0:
            logger.info("DIT_FAST=1: blocks compiled as whole graphs (models/fast_dit.py)")
    else:
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

    # ── Prompt (encoded on T5_RANK at load time), broadcast to all ──

    # All ranks: allocate buffers on device (same operation everywhere)
    ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == T5_RANK:
        # Pad to fixed size [1, 512, 4096] for broadcast
        for buf, c in zip((ctx_tensor, ctx_null_tensor), t5_out):
            padded = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
            padded[0, :c.shape[0]] = c
            buf.copy_(padded.to(NEURON_DEVICE))
        del t5_out

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)
    context = [ctx_tensor]
    context_null = [ctx_null_tensor]

    if rank == 0:
        logger.info("Prompt encoded and broadcast to all ranks")
        logger.info(f"  T5 context shape:  {ctx_tensor.shape} (dtype={ctx_tensor.dtype})")

    # ── Prepare image and noise ──
    import torchvision.transforms.functional as TF

    image_path = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")
    img = Image.open(image_path).convert("RGB")

    vae_stride = config.vae_stride  # (4, 16, 16)
    patch_size = config.patch_size   # (1, 2, 2)
    frame_num = FRAME_NUM
    assert (frame_num - 1) % vae_stride[0] == 0, f"FRAME_NUM must be 4k+1, got {frame_num}"
    max_area = 480 * 832

    ih, iw = img.height, img.width
    dh = patch_size[1] * vae_stride[1]
    dw = patch_size[2] * vae_stride[2]
    if OUT_HEIGHT and OUT_WIDTH:
        # Generate at the next multiple of 32 (VAE 16x * patch 2), center-crop after decode.
        oh = -(-OUT_HEIGHT // dh) * dh
        ow = -(-OUT_WIDTH // dw) * dw
        crop_h, crop_w = OUT_HEIGHT, OUT_WIDTH
    else:
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        crop_h, crop_w = oh, ow

    def crop_video(v):
        """Center-crop a decoded [C, F, H, W] video to the requested output size."""
        top, left = (v.shape[-2] - crop_h) // 2, (v.shape[-1] - crop_w) // 2
        return v[..., top:top + crop_h, left:left + crop_w]

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
        # VAE encode on Neuron — input must be bf16 to match model weights
        img_neuron = img_tensor.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        z = vae.encode([img_neuron])
        z_result = z[0].to(torch.bfloat16).contiguous()
        if z_result.device != NEURON_DEVICE:
            z_result = z_result.to(NEURON_DEVICE)
        z_img_device.copy_(z_result)

    dist.broadcast(z_img_device, src=VAE_RANK)

    if rank == 0:
        logger.info("-" * 70)
        logger.info("  INPUT SHAPES & MODEL CONFIG")
        logger.info("-" * 70)
        logger.info(f"  Image:           {oh}x{ow} (original {ih}x{iw}), output crop {crop_h}x{crop_w}")
        logger.info(f"  Frames:          {frame_num}")
        logger.info(f"  VAE stride:      {vae_stride}")
        logger.info(f"  Patch size:      {patch_size}")
        logger.info(f"  Latent shape:    [{z_dim}, {T_latent}, {H_latent}, {W_latent}]")
        logger.info(f"  VAE encode out:  {list(z_img_device.shape)} (dtype={z_img_device.dtype})")
        logger.info(f"  Noise shape:     [{z_dim}, {T_latent}, {H_latent}, {W_latent}]")
        logger.info(f"  Seq len (DiT):   {seq_len}")
        logger.info(f"  DiT config:      dim={config.dim}, heads={config.num_heads}, layers={config.num_layers}, ffn_dim={config.ffn_dim}")
        logger.info(f"  Heads/rank (TP): {config.num_heads // TP_DEGREE}")
        logger.info("-" * 70)

    # All ranks: move noise to neuron (same dtype everywhere)
    noise = noise.to(torch.bfloat16).to(NEURON_DEVICE)
    mask1, mask2 = masks_like([noise.cpu()], zero=True)
    mask2_device = [m.to(torch.bfloat16).to(NEURON_DEVICE) for m in mask2]

    # Build latent: image occupies temporal position 0, noise fills the rest
    # z_img_device is [z_dim, 1, H', W'], noise is [z_dim, T_latent, H', W']
    latent = noise.clone()
    latent = (1. - mask2_device[0]) * z_img_device.expand_as(noise) + mask2_device[0] * latent

    NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
    NUM_RUNS = int(os.environ.get("NUM_RUNS", "2"))

    # Model expects context as list of 2D tensors [text_len, hidden_dim] (one per batch item)
    # Our tensors are [1, 512, 4096] — squeeze batch dim to get [512, 4096]
    arg_c = {'context': [context[0][0]], 'seq_len': seq_len}
    arg_null = {'context': [context_null[0][0]], 'seq_len': seq_len}

    fast_dit = None
    if full_cross is not None:
        # Moved only now, after T5 was freed: rank T5_RANK shares an HBM stack with a neighbour.
        full_cross = [fc.to(NEURON_DEVICE).eval().requires_grad_(False) for fc in full_cross]
    if full_o is not None:
        full_o = [fo.to(NEURON_DEVICE).eval().requires_grad_(False) for fo in full_o]
    if DIT_FAST:
        from models.fast_dit import FastWanDiT
        grid = (T_latent, H_latent // patch_size[1], W_latent // patch_size[2])
        assert grid[0] * grid[1] * grid[2] == seq_len, f"grid {grid} != seq_len {seq_len}"
        fast_dit = FastWanDiT(model, grid=grid, guide_scale=config.sample_guide_scale,
                              device=NEURON_DEVICE, cfg_group=cfg_group, pair_group=pair_group,
                              full_cross=full_cross, full_o=full_o)
        fast_dit.set_context(context[0], context_null[0])

    # Save original latent for re-use across runs
    latent_orig = latent.clone()
    if RUN_HASH:
        import hashlib
        def _h(t):
            return hashlib.md5(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]
    run_results = []

    for run_idx in range(NUM_RUNS):
        run_label = f"Run {run_idx+1}/{NUM_RUNS}"
        if rank == 0:
            logger.info(f"═══ {run_label} ({NUM_STEPS} steps) ═══")

        # Reset latent to original for each run
        latent = latent_orig.clone()

        # ── Denoising loop ──
        if DEVICE_SCHEDULER:
            from models.device_unipc import DeviceFlowUniPC as _Sched
        else:
            _Sched = FlowUniPCMultistepScheduler
        sample_scheduler = _Sched(
            num_train_timesteps=config.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(NUM_STEPS, device=torch.device("cpu"), shift=config.sample_shift)
        timesteps = sample_scheduler.timesteps

        dump_step0 = None
        run_start = time.time()
        denoise_start = time.time()
        for step_idx, t in enumerate(tqdm(timesteps, disable=(rank != 0), desc=run_label)):
            if fast_dit is not None:
                noise_pred = fast_dit(latent, t)
            else:
                latent_model_input = [latent]

                t_device = t.to(NEURON_DEVICE)
                temp_ts = (mask2_device[0][0][:, ::2, ::2] * t_device).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_device])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]

                guide_scale = config.sample_guide_scale
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            if DEVICE_SCHEDULER:
                if step_idx == 0 and DUMP_DIR and run_idx == NUM_RUNS - 1:
                    dump_step0 = (latent.cpu(), noise_pred.cpu(), float(t))
                latent = sample_scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                                               return_dict=False)[0].squeeze(0)
                latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent
                continue
            # Scheduler step on CPU
            if STEP_TIMING:
                t_s0 = time.perf_counter()
            noise_pred_cpu = noise_pred.cpu()
            if STEP_TIMING:
                t_s1 = time.perf_counter()
            if step_idx == 0 and DUMP_DIR and run_idx == NUM_RUNS - 1:
                dump_step0 = (latent.cpu(), noise_pred_cpu, float(t))  # same input in every config
            temp_x0 = sample_scheduler.step(
                noise_pred_cpu.unsqueeze(0), t,
                latent.cpu().unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0).to(NEURON_DEVICE)
            latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent
            if STEP_TIMING and rank == 0:
                # wait = device still running the DiT when the host asked for the result;
                # host = D2H copy + CPU scheduler + H2D copy + mask blend enqueue
                t_s2 = time.perf_counter()
                logger.info(f"[step {step_idx}] wait for DiT {1e3 * (t_s1 - t_s0):.1f}ms, "
                            f"host (D2H + scheduler + H2D) {1e3 * (t_s2 - t_s1):.1f}ms")

        if DEVICE_SCHEDULER:
            # Nothing in the loop waits for the device any more; wait here so the denoise and VAE
            # times are each phase's own (the host would otherwise be several steps ahead).
            torch.neuron.synchronize()
        denoise_time = time.time() - denoise_start
        if rank == 0:
            logger.info(f"{run_label} denoising: {denoise_time:.1f}s ({denoise_time/NUM_STEPS:.2f}s/step)")
            if fast_dit is not None and run_idx == 0:
                fast_dit.log_graph_breaks()

        # ── Decode with VAE ──
        dump_video = None
        if MEM_LOG:
            print(f"[mem] rank {rank} run {run_idx + 1} before VAE: {device_mem_str()}", flush=True)
        vae_start = time.time()
        if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
            # VAE TP decode: all VAE TP ranks decode together (sharded channels)
            if rank == VAE_TP_RANKS[0]:
                logger.info(f"{run_label} VAE decode with TP={VAE_TP_DEGREE}...")
            x0 = [latent.to(torch.bfloat16)]
            videos = vae.decode(x0)
            # Only rank 0 saves the video (output is all-reduced, same on all VAE ranks)
            if rank == VAE_RANK:
                video = crop_video(videos[0])
                import imageio
                output_path = "/tmp/wan2_ti2v_output.mp4"
                video_cpu = video.cpu().float()
                video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
                if video_np.dim() == 4 and video_np.shape[0] == 3:
                    video_np = video_np.permute(1, 2, 3, 0)  # [F, H, W, C]
                frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
                imageio.mimwrite(output_path, frames, fps=24, codec='libx264', **MP4_KW)
                logger.info(f"Video saved to {output_path} ({len(frames)} frames)")
        elif VAE_TILED and VAE_STREAM:
            # Streaming tiled decode: rank 0 blends and encodes each latent-frame chunk while
            # the ranks are still decoding the rest (models/vae_tiled.decode_tiled_stream).
            from models.vae_tiled import decode_tiled_stream, decode_strips_stream
            import imageio
            output_path = "/tmp/wan2_ti2v_output.mp4"

            def _to_frames(acc):  # same truncation as the non-streaming branch below
                return crop_video(acc).add_(1).mul_(127.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()

            def _writer():
                return imageio.get_writer(output_path, fps=24, codec='libx264', **MP4_KW)

            if VAE_STRIPS:
                video_np = decode_strips_stream(vae, latent.to(torch.bfloat16), cpu_group, _writer,
                                                _to_frames)
            else:
                video_np = decode_tiled_stream(vae, latent.to(torch.bfloat16), cpu_group,
                                               VAE_TILE_OVERLAP, _writer, _to_frames)
            if rank == 0:
                dump_video = video_np
                logger.info(f"Video saved to {output_path} ({len(video_np)} frames, streamed)")
        elif VAE_TILED:
            # Tiled decode across all ranks; rank 0 receives the blended video
            from models.vae_tiled import decode_tiled, tile_grid
            if rank == 0 and run_idx == 0:
                th, tw, ys, xs = tile_grid(latent.shape[2], latent.shape[3], world_size, VAE_TILE_OVERLAP)
                logger.info(f"VAE tiled decode: {len(ys)}x{len(xs)} tiles of {th}x{tw} latent "
                            f"(overlap {VAE_TILE_OVERLAP}) over {world_size} ranks")
            print(f"[mem] rank {rank} before VAE: {device_mem_str()}", flush=True)
            video = decode_tiled(vae, latent.to(torch.bfloat16), cpu_group, VAE_TILE_OVERLAP)
            print(f"[mem] rank {rank} after VAE: {device_mem_str()}", flush=True)
            if rank == 0:
                t_w = time.time()
                import imageio
                output_path = "/tmp/wan2_ti2v_output.mp4"
                # video is float32 in [-1, 1] (clamped in decode_tiled); same truncation as
                # ((v * 0.5 + 0.5) * 255).byte(), without the extra full-size temporaries.
                video_np = crop_video(video).add_(1).mul_(127.5).to(torch.uint8)
                video_np = video_np.permute(1, 2, 3, 0).contiguous().numpy()  # [F, H, W, C]
                dump_video = video_np
                t_c = time.time()
                imageio.mimwrite(output_path, list(video_np), fps=24, codec='libx264', **MP4_KW)
                logger.info(f"Video saved to {output_path} ({len(video_np)} frames); "
                            f"convert {t_c - t_w:.2f}s + mp4 write {time.time() - t_c:.2f}s")
        elif rank == VAE_RANK:
            # Original single-rank decode with NKI kernels
            logger.info(f"{run_label} VAE decode on Neuron...")
            x0 = [latent.to(torch.bfloat16)]
            videos = vae.decode(x0)
            video = crop_video(videos[0])
            import imageio
            output_path = "/tmp/wan2_ti2v_output.mp4"
            video_cpu = video.cpu().float()
            video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
            if video_np.dim() == 4 and video_np.shape[0] == 3:
                video_np = video_np.permute(1, 2, 3, 0)  # [F, H, W, C]
            frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
            imageio.mimwrite(output_path, frames, fps=24, codec='libx264', **MP4_KW)
            logger.info(f"Video saved to {output_path} ({len(frames)} frames)")

        dist.barrier()
        vae_time = time.time() - vae_start
        if MEM_LOG:
            print(f"[mem] rank {rank} run {run_idx + 1} after VAE: {device_mem_str()}", flush=True)
        run_time = time.time() - run_start

        if rank == 0 and DUMP_DIR and run_idx == NUM_RUNS - 1 and dump_video is not None:
            _dump_outputs(DUMP_DIR, latent, dump_video)  # after the timers
            torch.save({"latent_in": dump_step0[0].float(), "noise_pred": dump_step0[1].float(),
                        "t": dump_step0[2], "ctx": context[0][0].float().cpu(),
                        "ctx_null": context_null[0][0].float().cpu()},
                       os.path.join(DUMP_DIR, "step0.pt"))
        if rank == 0:
            logger.info(f"═══ {run_label} DONE: denoise={denoise_time:.1f}s, vae={vae_time:.1f}s, total={run_time:.1f}s ═══")
            run_results.append({
                'run': run_idx + 1,
                'denoise': denoise_time,
                'vae': vae_time,
                'total': run_time,
            })
        if RUN_HASH:
            lh = _h(latent)
            mh = ""
            if rank == 0 and os.path.exists("/tmp/wan2_ti2v_output.mp4"):
                with open("/tmp/wan2_ti2v_output.mp4", "rb") as fh:
                    mh = " mp4=" + hashlib.md5(fh.read()).hexdigest()[:12]
                if os.environ.get("RUN_HASH_KEEP"):  # keep each run's mp4: <prefix>_run<i>.mp4
                    import shutil
                    shutil.copy("/tmp/wan2_ti2v_output.mp4",
                                f"{os.environ['RUN_HASH_KEEP']}_run{run_idx + 1}.mp4")
            print(f"[hash] rank {rank} run {run_idx + 1} latent={lh}{mh}", flush=True)

    if RUN_HASH:  # once-per-process state, hashed after the runs (no extra syncs before them)
        print(f"[hash] rank {rank} inputs ctx={_h(context[0])} ctx_null={_h(context_null[0])} "
              f"z_img={_h(z_img_device)} latent0={_h(latent_orig)}", flush=True)
        # Once-per-process device state: every block's cross-attention K/V, and per-parameter
        # fp32 sums of the device weights (DiT, unsharded cross-attention copies, VAE).
        if fast_dit is not None:
            kv = torch.cat([t.float().flatten().cpu() for pair in fast_dit.context for t in pair])
            print(f"[hash] rank {rank} ctxkv={_h(kv)}", flush=True)
        mods = [("dit", model)] + [("xca", m) for m in (full_cross or [])]
        if locals().get("vae") is not None:
            mods.append(("vae", vae.model))
        for tag in ("dit", "xca", "vae"):
            ps = [p for t, m in mods if t == tag for p in m.parameters() if p.numel() > 0]
            if ps:
                sums = torch.stack([p.detach().float().sum() for p in ps]).cpu()
                print(f"[hash] rank {rank} wsum_{tag}={_h(sums)} (n={len(ps)})", flush=True)
    # ── Final benchmark summary ──
    if rank == 0 and run_results:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Instance:        {INSTANCE_TYPE}")
        logger.info(f"  LNC config:      {LNC_CONFIG}")
        logger.info(f"  NeuronDevices:   {NUM_NEURON_DEVICES}")
        logger.info(f"  TP degree:       {TP_DEGREE}  (DiT)  |  SP: {SP_DEGREE}  |  VAE TP: {VAE_TP_DEGREE}")
        logger.info(f"  NKI kernels:     DiT={USE_NKI_KERNELS}, VAE={USE_NKI_VAE}, NST_self_attn={USE_NST_SELF_ATTN}")
        logger.info(f"  Resolution:      {oh}x{ow} generated, {crop_h}x{crop_w} output, {frame_num} frames")
        logger.info(f"  Latent:          [{z_dim}, {T_latent}, {H_latent}, {W_latent}]")
        logger.info(f"  Seq len:         {seq_len}")
        logger.info(f"  Steps:           {NUM_STEPS}")
        logger.info("-" * 70)
        logger.info(f"  {'Run':<6} {'Denoise':>10} {'s/step':>10} {'VAE dec':>10} {'Total':>10}")
        logger.info("-" * 70)
        for r in run_results:
            s_per_step = r['denoise'] / NUM_STEPS
            logger.info(f"  {r['run']:<6} {r['denoise']:>9.1f}s {s_per_step:>9.1f}s {r['vae']:>9.1f}s {r['total']:>9.1f}s")
        # Warm run = last run (compilation done)
        warm = run_results[-1]
        logger.info("-" * 70)
        logger.info(f"  Warm run:  denoise={warm['denoise']:.1f}s ({warm['denoise']/NUM_STEPS:.1f}s/step), vae={warm['vae']:.1f}s, total={warm['total']:.1f}s")
        logger.info("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
