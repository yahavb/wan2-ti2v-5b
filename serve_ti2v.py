"""Wan2.2-TI2V-5B FastAPI server on one Trainium2 device (8 ranks, torchrun).

Serves the same compiled pipeline the benchmark measures (inference_neuron_ti2v.py with DIT_FAST=1):
block-compiled DiT (models/fast_dit.py) with its env flags (FAST_ATTN_IMPL, CFG_PARALLEL,
CFG_GROUPS, FAST_SP, FAST_LOCAL_CROSS, ...), the VAE tiled over every rank (VAE_TILED, VAE_STREAM)
with the nkilib conv kernels, and the UniPC step on the device (DEVICE_SCHEDULER).

Serving-specific choices:
  - Fixed output size (SERVE_HEIGHT x SERVE_WIDTH; default: the warm-up image's size). Every
    request image is resized and center-cropped to it, so no request compiles a new graph.
  - T5 on the CPU of T5_RANK (T5_DEVICE=cpu, default). The benchmark frees the on-device T5 after
    one prompt; a server needs it for every request, and its ~11 GB does not fit next to the DiT
    and VAE on that rank's HBM stack. The negative prompt is encoded once; prompts are cached.
  - WARMUP_RUNS (default 3) full runs before /readiness turns ready: on the current image run 2
    still traces NKI kernels, only run 3 runs at steady state.
  - The HTTP server starts first, so /health answers during loading and warm-up.
  - All Neuron collectives run on the main thread (the thread that compiled the graphs). FastAPI
    hands requests to it through a queue.

API:
  POST /generate  — {"prompt": "...", "image_url": "..." | "image_base64": "...", "seed": 42}
                    Returns {"video": "<base64 mp4>", "execution_time": 8.7, "frames": 81, ...}
  GET  /health    — {"status": "ok"}
  GET  /readiness — {"status": "ready"} after warm-up, else 503
"""
import base64
import collections
import io
import logging
import os
import queue
import sys
import threading
import time

import torch
import torch.distributed as dist
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True,
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_mlir']:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

WAN_DIR = os.environ.get("WAN_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAN_DIR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "8"))
T5_RANK = int(os.environ.get("T5_RANK", "2"))
T5_DEVICE = os.environ.get("T5_DEVICE", "cpu")  # "cpu" or "neuron"
T5_CPU_THREADS = int(os.environ.get("T5_CPU_THREADS", "32"))
PROMPT_CACHE = int(os.environ.get("PROMPT_CACHE", "32"))
VAE_RANK = 0
VAE_TILED = os.environ.get("VAE_TILED", "1") == "1"
VAE_TILE_OVERLAP = int(os.environ.get("VAE_TILE_OVERLAP", "4"))
VAE_STREAM = os.environ.get("VAE_STREAM", "1") == "1"
CFG_PARALLEL = os.environ.get("CFG_PARALLEL", "1") == "1"
CFG_GROUPS = os.environ.get("CFG_GROUPS", "interleave")
DEVICE_SCHEDULER = os.environ.get("DEVICE_SCHEDULER", "1") == "1"
if os.environ.get("MP4_VFORK", "1") == "1":  # spawn ffmpeg with vfork (see inference_neuron_ti2v.py)
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
MP4_KW = dict(quality=None, ffmpeg_params=["-preset", os.environ.get("MP4_PRESET", "veryfast"),
                                           "-crf", os.environ.get("MP4_CRF", "20")])
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
FRAME_NUM = int(os.environ.get("FRAME_NUM", "81"))
SERVE_HEIGHT = int(os.environ.get("SERVE_HEIGHT", "0"))
SERVE_WIDTH = int(os.environ.get("SERVE_WIDTH", "0"))
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "3"))
PORT = int(os.environ.get("PORT", "8000"))

NEURON_DEVICE = torch.device("neuron")
INPUTS_PATH = "/tmp/current_serve_inputs.pt"
RESULT_PATH = "/tmp/current_serve_result.mp4"
WARMUP_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, "
    "distant green hills, and a blue sky dotted with white clouds."
)
WARMUP_IMAGE = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")

_state = {"ready": False, "phase": "starting"}


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def load_image(src):
    """PIL RGB image from a path, an http(s) URL or raw bytes."""
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    if src.startswith("http"):
        import urllib.request
        with urllib.request.urlopen(src) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(src).convert("RGB")


def fit_image(img, oh, ow):
    """Resize to cover (oh, ow) and center-crop, as the benchmark does; [C, 1, H, W] in [-1, 1]."""
    import torchvision.transforms.functional as TF
    scale = max(ow / img.width, oh / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    return TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(1)


class Pipeline:
    """Loads and compiles everything once at a fixed shape; generate() runs on every rank."""

    def __init__(self, rank, world_size):
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE
        from wan.utils.utils import best_output_size, masks_like
        from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank
        from models.fast_dit import FastWanDiT

        self.rank, self.world_size = rank, world_size
        self.config = config = WAN_CONFIGS['ti2v-5B']
        self.cpu_group = dist.new_group(backend="gloo") if VAE_TILED else None
        global T5_RANK
        T5_RANK = min(T5_RANK, world_size - 1)

        # Fixed generation shape: from SERVE_HEIGHT/WIDTH, else the warm-up image's aspect.
        vs, ps = config.vae_stride, config.patch_size
        dh, dw = ps[1] * vs[1], ps[2] * vs[2]
        if SERVE_HEIGHT and SERVE_WIDTH:
            self.oh, self.ow = -(-SERVE_HEIGHT // dh) * dh, -(-SERVE_WIDTH // dw) * dw
        else:
            img = load_image(WARMUP_IMAGE)
            self.ow, self.oh = best_output_size(img.width, img.height, dw, dh, 480 * 832)
        assert (FRAME_NUM - 1) % vs[0] == 0, f"FRAME_NUM must be 4k+1, got {FRAME_NUM}"
        self.z_dim = config.vae_dim[0] if hasattr(config, 'vae_dim') else 48
        self.T_lat = (FRAME_NUM - 1) // vs[0] + 1
        self.H_lat, self.W_lat = self.oh // vs[1], self.ow // vs[2]
        grid = (self.T_lat, self.H_lat // ps[1], self.W_lat // ps[2])

        # T5: kept for every request, on T5_RANK
        self.text_encoder = None
        self._ctx_cache = collections.OrderedDict()
        if rank == T5_RANK:
            if T5_DEVICE == "cpu":
                torch.set_num_threads(T5_CPU_THREADS)
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len, dtype=config.t5_dtype, device=torch.device('cpu'),
                checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
                tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
            if T5_DEVICE != "cpu":
                self.text_encoder.model = self.text_encoder.model.to(NEURON_DEVICE)
            logger.info(f"T5 on rank {T5_RANK} ({T5_DEVICE})")
        elif rank == 0:
            # torchrun sets OMP_NUM_THREADS=1; rank 0 blends the VAE tiles and converts the video.
            torch.set_num_threads(int(os.environ.get("RANK0_CPU_THREADS", "16")))

        # VAE: every rank when tiled (each decodes its tiles), else rank 0
        self.vae = None
        if rank == VAE_RANK or VAE_TILED:
            self.vae = Wan2_2_VAE(vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
                                  device=torch.device('cpu'))
            self.vae.model = self.vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
            self.vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
                              if isinstance(s, torch.Tensor) else s for s in self.vae.scale]
            n_packed = self.vae.prepack_nki_filters()
            if rank == 0:
                logger.info(f"VAE on Neuron ({n_packed} decoder convs pre-packed for nkilib)")

        # DiT: TP-sharded; CFG-parallel splits the ranks into a cond and an uncond TP group
        model = WanModel.from_pretrained(MODEL_PATH)
        model.eval().requires_grad_(False)
        dit_tp = TP_DEGREE // 2 if CFG_PARALLEL else TP_DEGREE
        interleave = CFG_PARALLEL and CFG_GROUPS == "interleave"
        tp_groups = ([list(range(0, world_size, 2)), list(range(1, world_size, 2))]
                     if interleave else None)
        init_tp_group(tp_degree=dit_tp, groups=tp_groups)
        full_cross = None
        if CFG_PARALLEL and os.environ.get("FAST_SP", "0") == "1" \
                and os.environ.get("FAST_LOCAL_CROSS", "0") == "1":
            import copy
            full_cross = [copy.deepcopy(b.cross_attn).to(torch.bfloat16) for b in model.blocks]
        shard_model_tp(model, get_tp_rank(), dit_tp)
        cfg_group, pair_group = None, None
        if CFG_PARALLEL:
            assert world_size == TP_DEGREE and TP_DEGREE % 2 == 0, "CFG_PARALLEL needs an even world size"
            if interleave:
                cfg_group, pairs = rank % 2, [[2 * i, 2 * i + 1] for i in range(dit_tp)]
            else:
                cfg_group, pairs = rank // dit_tp, [[i, i + dit_tp] for i in range(dit_tp)]
            for p in pairs:  # every rank creates every pair group, same order
                g = dist.new_group(p)
                if rank in p:
                    pair_group = g
        if full_cross is not None:
            for b in model.blocks:  # the sharded cross-attention is unused in this mode
                for lin_ in (b.cross_attn.q, b.cross_attn.k, b.cross_attn.v, b.cross_attn.o):
                    lin_.weight = torch.nn.Parameter(lin_.weight.new_empty(0), requires_grad=False)
            full_cross = [fc.to(NEURON_DEVICE).eval().requires_grad_(False) for fc in full_cross]
        model = model.to(torch.bfloat16).to(NEURON_DEVICE)
        self.dit = FastWanDiT(model, grid=grid, guide_scale=config.sample_guide_scale,
                              device=NEURON_DEVICE, cfg_group=cfg_group, pair_group=pair_group,
                              full_cross=full_cross)

        _, mask2 = masks_like([torch.zeros(self.z_dim, self.T_lat, self.H_lat, self.W_lat)], zero=True)
        self.mask2 = mask2[0].to(torch.bfloat16).to(NEURON_DEVICE)
        self.ctx_null = self._encode(config.sample_neg_prompt)  # same for every request
        dist.barrier()
        if rank == 0:
            logger.info(f"Pipeline loaded: {self.oh}x{self.ow}, {FRAME_NUM} frames, grid {grid}, "
                        f"T5 {T5_DEVICE}, CFG_PARALLEL={CFG_PARALLEL} ({CFG_GROUPS}), "
                        f"VAE tiled={VAE_TILED} stream={VAE_STREAM}")

    def _encode(self, prompt):
        """T5 on T5_RANK (cached by prompt), broadcast to every rank: [1, 512, 4096] bf16."""
        ctx = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        if self.rank == T5_RANK:
            c = self._ctx_cache.pop(prompt, None)
            if c is None:
                dev = torch.device("cpu") if T5_DEVICE == "cpu" else NEURON_DEVICE
                c = self.text_encoder([prompt], dev)[0].to(torch.bfloat16).cpu()
            self._ctx_cache[prompt] = c
            while len(self._ctx_cache) > PROMPT_CACHE:
                self._ctx_cache.popitem(last=False)
            ctx[0, :c.shape[0]] = c.to(NEURON_DEVICE)
        dist.broadcast(ctx, src=T5_RANK)
        return ctx

    def generate(self, prompt, image, seed, out_path=RESULT_PATH):
        """Every rank calls this with the same seed. prompt is read on T5_RANK, image on rank 0.
        Returns (denoise_s, vae_s, total_s, n_frames) on rank 0."""
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from models.device_unipc import DeviceFlowUniPC
        config = self.config
        t0 = time.time()

        z_img = torch.zeros(self.z_dim, 1, self.H_lat, self.W_lat, dtype=torch.bfloat16,
                            device=NEURON_DEVICE)
        if self.rank == VAE_RANK:
            img = fit_image(load_image(image), self.oh, self.ow)
            z_img.copy_(self.vae.encode([img.to(device=NEURON_DEVICE, dtype=torch.bfloat16)])[0]
                        .to(torch.bfloat16))
        dist.broadcast(z_img, src=VAE_RANK)
        self.dit.set_context(self._encode(prompt), self.ctx_null)

        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(seed)
        noise = torch.randn(self.z_dim, self.T_lat, self.H_lat, self.W_lat, dtype=torch.float32,
                            generator=g, device=torch.device("cpu"))
        m = self.mask2
        latent = noise.to(torch.bfloat16).to(NEURON_DEVICE)
        latent = (1. - m) * z_img.expand_as(latent) + m * latent

        sched = (DeviceFlowUniPC if DEVICE_SCHEDULER else FlowUniPCMultistepScheduler)(
            num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sched.set_timesteps(NUM_STEPS, device=torch.device("cpu"), shift=config.sample_shift)
        td = time.time()
        for t in sched.timesteps:
            noise_pred = self.dit(latent, t)
            if DEVICE_SCHEDULER:
                latent = sched.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                                    return_dict=False)[0].squeeze(0)
            else:
                latent = sched.step(noise_pred.cpu().unsqueeze(0), t, latent.cpu().unsqueeze(0),
                                    return_dict=False, generator=g)[0].squeeze(0).to(NEURON_DEVICE)
            latent = (1. - m) * z_img.expand_as(latent) + m * latent
        if DEVICE_SCHEDULER:
            torch.neuron.synchronize()
        denoise = time.time() - td

        tv = time.time()
        n_frames = self._decode(latent.to(torch.bfloat16), out_path)
        dist.barrier()
        return denoise, time.time() - tv, time.time() - t0, n_frames

    def _decode(self, latent, out_path):
        import imageio

        def to_frames(v):  # float [-1, 1] [C, F, H, W] -> uint8 [F, H, W, C]
            return v.add_(1).mul_(127.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()

        if VAE_TILED and VAE_STREAM:
            from models.vae_tiled import decode_tiled_stream
            frames = decode_tiled_stream(
                self.vae, latent, self.cpu_group, VAE_TILE_OVERLAP,
                lambda: imageio.get_writer(out_path, fps=24, codec='libx264', **MP4_KW), to_frames)
            return len(frames) if self.rank == 0 else 0
        if VAE_TILED:
            from models.vae_tiled import decode_tiled
            video = decode_tiled(self.vae, latent, self.cpu_group, VAE_TILE_OVERLAP)
        elif self.rank == VAE_RANK:
            video = self.vae.decode([latent])[0]
        if self.rank != 0:
            return 0
        frames = to_frames(video)
        imageio.mimwrite(out_path, list(frames), fps=24, codec='libx264', **MP4_KW)
        return len(frames)


def start_http(request_q, response_q):
    """Rank 0: FastAPI in a daemon thread. Requests go to the main thread through request_q."""
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional

    app = FastAPI(title="Wan2.2-TI2V-5B on Trainium (native PyTorch)")

    class GenerateRequest(BaseModel):
        prompt: str
        image_url: Optional[str] = None
        image_base64: Optional[str] = None
        seed: int = 42

    @app.get("/health")
    def health():
        return {"status": "ok", "phase": _state["phase"]}

    @app.get("/readiness")
    def readiness():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["phase"])
        return {"status": "ready"}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail="Model not ready")
        if not req.image_url and not req.image_base64:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 is required")
        image = base64.b64decode(req.image_base64) if req.image_base64 else req.image_url
        logger.info(f"[REQUEST] prompt='{req.prompt[:80]}' seed={req.seed}")
        request_q.put({"prompt": req.prompt, "image": image, "seed": req.seed})
        try:
            result = response_q.get(timeout=600)
        except queue.Empty:
            raise HTTPException(status_code=504, detail="Inference timed out")
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail=str(result))
        return result

    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info"),
                     daemon=True).start()
    logger.info(f"[SERVER] FastAPI on port {PORT} (/health live, /readiness after warm-up)")


def main():
    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)
    request_q, response_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    if rank == 0:
        start_http(request_q, response_q)
        _state["phase"] = "loading"

    pipe = Pipeline(rank, world_size)

    # Warm up until steady state: run 1 compiles/loads, run 2 can still trace NKI kernels.
    if rank == 0:
        _state["phase"] = "warming up"
    for i in range(WARMUP_RUNS):
        d, v, tot, _ = pipe.generate(WARMUP_PROMPT, WARMUP_IMAGE, seed=42)
        if rank == 0:
            logger.info(f"[WARMUP] run {i + 1}/{WARMUP_RUNS}: denoise={d:.1f}s vae={v:.1f}s total={tot:.1f}s")
    dist.barrier()
    if rank == 0:
        pipe.dit.log_graph_breaks()
        _state.update(ready=True, phase="ready")
        logger.info("[READY] warmed up, accepting requests")

    # Request loop. Rank 0 broadcasts a signal (from the main thread, like every collective);
    # T5_RANK reads the prompt from INPUTS_PATH, rank 0 already holds the image.
    while True:
        req = None
        if rank == 0:
            req = request_q.get()
            torch.save({"prompt": req["prompt"]}, INPUTS_PATH)
            signal = torch.tensor([req["seed"], 1], dtype=torch.long).to(NEURON_DEVICE)
        else:
            signal = torch.zeros(2, dtype=torch.long).to(NEURON_DEVICE)
        dist.broadcast(signal, src=0)
        seed, action = (int(x) for x in signal.cpu().tolist())
        if action != 1:
            continue
        try:
            prompt = req["prompt"] if rank == 0 else torch.load(INPUTS_PATH)["prompt"]
            d, v, tot, n = pipe.generate(prompt, req["image"] if rank == 0 else None, seed)
            if rank == 0:
                with open(RESULT_PATH, "rb") as f:
                    video = f.read()
                logger.info(f"[RESPONSE] total={tot:.1f}s denoise={d:.1f}s vae={v:.1f}s "
                            f"{n} frames {len(video)} bytes")
                response_q.put({"video": base64.b64encode(video).decode(), "execution_time": round(tot, 2),
                                "denoise_time": round(d, 2), "vae_time": round(v, 2), "frames": n})
        except Exception as e:
            logger.error(f"[rank {rank}] request failed: {e}")
            if rank == 0:
                response_q.put(e)


if __name__ == "__main__":
    main()
