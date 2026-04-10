import os
import sys
import gc
import io
import random
import base64
import tempfile
import traceback

import torch
import numpy as np
import runpod
from PIL import Image, ImageOps

from diffusers import WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/wan2.2")
GGUF_DIR = os.environ.get("GGUF_DIR", "/models/gguf")
LORA_DIR = os.environ.get("LORA_DIR", "/models/loras")
QUANT_METHOD = os.environ.get("QUANT_METHOD", "gguf")  # gguf, bnb4, bnb8, none
ENABLE_SAGE_ATTENTION = os.environ.get("ENABLE_SAGE_ATTENTION", "true").lower() == "true"
CACHE_THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0"))
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
DEFAULT_FLOW_SHIFT = float(os.environ.get("DEFAULT_FLOW_SHIFT", "5.0"))
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "false").lower() == "true"

DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts, "
    "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, "
    "disfigured, malformed limbs, fused fingers, still image, messy background, "
    "three legs, many people in background, walking backwards"
)

# ---------------------------------------------------------------------------
# Globals (initialised once at startup)
# ---------------------------------------------------------------------------
pipe = None
available_loras: list[str] = []
mod_value = 16
_active_lora_key: tuple | None = None  # track (names, weights) to skip redundant switches


def load_pipeline():
    global pipe, available_loras, mod_value

    print("=" * 60)
    print("Sneaky Vidgen – Wan2.2 I2V")
    print(f"  model        {MODEL_PATH}")
    print(f"  quant        {QUANT_METHOD}")
    print(f"  sage_attn    {ENABLE_SAGE_ATTENTION}")
    print(f"  cache_thresh {CACHE_THRESHOLD}")
    print(f"  cpu_offload  {ENABLE_CPU_OFFLOAD}")
    print(f"  flow_shift   {DEFAULT_FLOW_SHIFT}")
    print(f"  torch.compile {TORCH_COMPILE}")
    print("=" * 60)

    # -- Download model if not present ----------------------------------------
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}, downloading …")
        from huggingface_hub import snapshot_download

        skip = (
            ["transformer/diffusion_pytorch_model*", "transformer_2/diffusion_pytorch_model*"]
            if QUANT_METHOD == "gguf"
            else None
        )
        snapshot_download(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            local_dir=MODEL_PATH,
            ignore_patterns=skip,
            token=os.environ.get("HF_TOKEN") or None,
        )

    # -- Load pipeline --------------------------------------------------------
    print(f"Loading pipeline ({QUANT_METHOD}) …")

    if QUANT_METHOD == "gguf":
        import glob
        from diffusers import WanTransformer3DModel, GGUFQuantizationConfig

        gguf_cfg = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        hn_files = glob.glob(os.path.join(GGUF_DIR, "HighNoise", "*.gguf"))
        ln_files = glob.glob(os.path.join(GGUF_DIR, "LowNoise", "*.gguf"))
        if not hn_files or not ln_files:
            raise FileNotFoundError(
                f"GGUF files not found in {GGUF_DIR}/HighNoise/ and {GGUF_DIR}/LowNoise/. "
                "Run builder.py or set GGUF_DIR correctly."
            )

        # Provide explicit Wan2.2 config — auto-detection wrongly maps to Wan2.1
        hn_config = os.path.join(MODEL_PATH, "transformer")
        ln_config = os.path.join(MODEL_PATH, "transformer_2")

        print(f"  HighNoise: {os.path.basename(hn_files[0])}")
        transformer = WanTransformer3DModel.from_single_file(
            hn_files[0],
            quantization_config=gguf_cfg,
            config=hn_config,
            torch_dtype=torch.bfloat16,
        )
        print(f"  LowNoise:  {os.path.basename(ln_files[0])}")
        transformer_2 = WanTransformer3DModel.from_single_file(
            ln_files[0],
            quantization_config=gguf_cfg,
            config=ln_config,
            torch_dtype=torch.bfloat16,
        )

        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_PATH,
            transformer=transformer,
            transformer_2=transformer_2,
            torch_dtype=torch.bfloat16,
        )

    elif QUANT_METHOD in ("bnb4", "bnb8"):
        from diffusers.quantizers import PipelineQuantizationConfig

        backend = "bitsandbytes_4bit" if QUANT_METHOD == "bnb4" else "bitsandbytes_8bit"
        qkw = (
            {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16}
            if QUANT_METHOD == "bnb4"
            else {"load_in_8bit": True}
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_PATH,
            quantization_config=PipelineQuantizationConfig(
                quant_backend=backend, quant_kwargs=qkw,
                components_to_quantize=["transformer", "transformer_2"],
            ),
            torch_dtype=torch.bfloat16,
        )

    else:
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

    # -- Device / offload strategy --------------------------------------------
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # -- Scheduler (flow-matching with configurable shift) --------------------
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=DEFAULT_FLOW_SHIFT
    )

    # -- Dimension alignment constant -----------------------------------------
    try:
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    except Exception:
        mod_value = 16

    # -- VAE memory optimisations (always beneficial) -------------------------
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    # -- SageAttention (INT8-quantised attention, ~2× speedup) ----------------
    if ENABLE_SAGE_ATTENTION:
        try:
            pipe.transformer.set_attention_backend("sage")
            if pipe.transformer_2 is not None:
                pipe.transformer_2.set_attention_backend("sage")
            print("SageAttention enabled")
        except Exception as exc:
            print(f"SageAttention unavailable: {exc}")

    # -- FirstBlockCache (TeaCache-style, opt-in for speed) -------------------
    if CACHE_THRESHOLD > 0:
        try:
            from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

            cfg = FirstBlockCacheConfig(threshold=CACHE_THRESHOLD)
            apply_first_block_cache(pipe.transformer, cfg)
            if pipe.transformer_2 is not None:
                apply_first_block_cache(pipe.transformer_2, cfg)
            print(f"FirstBlockCache threshold={CACHE_THRESHOLD}")
        except Exception as exc:
            print(f"FirstBlockCache unavailable: {exc}")

    # -- torch.compile (opt-in, NOT compatible with GGUF) ----------------------
    if TORCH_COMPILE:
        if QUANT_METHOD == "gguf":
            print("torch.compile skipped (incompatible with GGUF quantization)")
        else:
            try:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="max-autotune-no-cudagraphs"
                )
                if pipe.transformer_2 is not None:
                    pipe.transformer_2 = torch.compile(
                        pipe.transformer_2, mode="max-autotune-no-cudagraphs"
                    )
                print("torch.compile applied (first request will be slow)")
            except Exception as exc:
                print(f"torch.compile failed: {exc}")

    # -- LoRAs ----------------------------------------------------------------
    if os.path.exists(LORA_DIR):
        for lf in sorted(os.listdir(LORA_DIR)):
            if not lf.endswith(".safetensors"):
                continue
            name = os.path.splitext(lf)[0]
            try:
                pipe.load_lora_weights(
                    os.path.join(LORA_DIR, lf),
                    adapter_name=name,
                    load_into_transformer_2=True,
                )
                available_loras.append(name)
                print(f"Loaded LoRA: {name}")
            except Exception as exc:
                print(f"LoRA {lf} failed: {exc}")

        if available_loras:
            pipe.disable_lora()

    print("=" * 60)
    print(f"Ready. LoRAs: {available_loras}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers (zero-allocation hot-path)
# ---------------------------------------------------------------------------
def decode_image(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def align_dim(v: int) -> int:
    return max((v // mod_value) * mod_value, mod_value)


def align_frames(n: int) -> int:
    # Wan2.2 requires num_frames = 4k + 1
    if (n - 1) % 4 != 0:
        n = ((n - 1) // 4) * 4 + 1
    return max(n, 5)


def encode_video(frames, fps: int, fmt: str) -> bytes:
    fd, path = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(fd)
    try:
        export_to_video(frames, path, fps=fps)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    global _active_lora_key

    inp = job["input"]

    # -- Meta actions (no generation) -----------------------------------------
    action = inp.get("action")
    if action == "list_loras":
        return {"available_loras": available_loras}
    if action == "health":
        return {
            "status": "ok",
            "quantization": QUANT_METHOD,
            "sage_attention": ENABLE_SAGE_ATTENTION,
            "cache_threshold": CACHE_THRESHOLD,
            "flow_shift": DEFAULT_FLOW_SHIFT,
            "torch_compile": TORCH_COMPILE,
            "available_loras": available_loras,
        }

    # -- Validate -------------------------------------------------------------
    image_b64 = inp.get("image")
    if not image_b64:
        return {"error": "Missing required 'image' field (base64-encoded)."}

    # -- Parameters (high-quality defaults) -----------------------------------
    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", DEFAULT_NEGATIVE)
    width = align_dim(int(inp.get("width", 1280)))
    height = align_dim(int(inp.get("height", 720)))
    steps = int(inp.get("steps", 30))
    cfg = float(inp.get("cfg_scale", 3.5))
    num_frames = align_frames(int(inp.get("num_frames", 81)))
    fps = int(inp.get("fps", 16))
    seed = int(inp.get("seed", 0)) or random.randint(0, 2**32 - 1)
    output_format = inp.get("output_format", "mp4").lower()

    flow_shift = inp.get("flow_shift")  # None = use default

    # -- Parse LoRAs (array of {name, strength} or legacy single fields) ------
    loras_raw = inp.get("loras", [])
    if not loras_raw and inp.get("lora_name"):
        loras_raw = [{"name": inp["lora_name"], "strength": float(inp.get("lora_strength", 1.0))}]

    lora_names = []
    lora_weights = []
    for entry in loras_raw:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        strength = 1.0 if isinstance(entry, str) else float(entry.get("strength", 1.0))
        if name not in available_loras:
            return {"error": f"LoRA '{name}' not found. Available: {available_loras}"}
        lora_names.append(name)
        lora_weights.append(strength)

    lora_desc = ", ".join(f"{n}@{w}" for n, w in zip(lora_names, lora_weights))
    print(
        f"Job: {width}x{height} {num_frames}f {steps}steps cfg={cfg} seed={seed}"
        + (f" loras=[{lora_desc}]" if lora_names else "")
    )

    try:
        # -- LoRA switching (skip if unchanged) -------------------------------
        lora_key = (tuple(lora_names), tuple(lora_weights)) if lora_names else None
        if lora_key != _active_lora_key:
            if lora_names:
                pipe.enable_lora()
                pipe.set_adapters(lora_names, adapter_weights=lora_weights)
            elif available_loras:
                pipe.disable_lora()
            _active_lora_key = lora_key

        # -- Per-request flow shift override ----------------------------------
        saved_scheduler = None
        if flow_shift is not None:
            saved_scheduler = pipe.scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=float(flow_shift)
            )

        # -- Prepare input image ----------------------------------------------
        image = decode_image(image_b64)
        image = image.resize((width, height), Image.LANCZOS)

        # -- Generate ---------------------------------------------------------
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        frames = output.frames[0]

        # -- Restore scheduler ------------------------------------------------
        if saved_scheduler is not None:
            pipe.scheduler = saved_scheduler

        # -- Encode video -----------------------------------------------------
        video_bytes = encode_video(frames, fps, output_format)
        video_b64 = base64.b64encode(video_bytes).decode("ascii")

        del output, frames, image
        torch.cuda.empty_cache()

        print(f"Done. {len(video_bytes) / 1048576:.1f} MB seed={seed}")

        return {
            "video": video_b64,
            "video_format": output_format,
            "seed": seed,
            "params": {
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "num_frames": num_frames,
                "fps": fps,
                "quantization": QUANT_METHOD,
                "flow_shift": flow_shift or DEFAULT_FLOW_SHIFT,
                "loras": [{"name": n, "strength": w} for n, w in zip(lora_names, lora_weights)] or None,
            },
        }

    except Exception as exc:
        torch.cuda.empty_cache()
        traceback.print_exc()
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
load_pipeline()
runpod.serverless.start({"handler": handler})
