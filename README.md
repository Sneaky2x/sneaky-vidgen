# Sneaky Vidgen

RunPod serverless worker for **Wan2.2 Image-to-Video** generation using HuggingFace Diffusers.

Wan2.2 I2V-A14B is a Mixture-of-Experts model (27B total / 14B active) with two transformer stages: a high-noise expert handling 90% of denoising and a low-noise expert refining the final 10%.

## Features

- **Wan2.2 I2V-A14B** via `WanImageToVideoPipeline` (no ComfyUI dependency)
- **Quantization** -- BitsAndBytes NF4 (default, ~20 GB VRAM) or 8-bit (~40 GB) or full BF16 (~66 GB)
- **Multiple LoRAs** -- stack any combination with per-LoRA strength control (FusionX, Lightx2v, CivitAI, etc.)
- **SageAttention** -- INT8 quantised attention for ~2x speedup with no quality loss
- **FirstBlockCache** -- TeaCache-style block caching (opt-in for speed)
- **torch.compile** -- opt-in JIT compilation for additional throughput
- **VAE tiling + slicing** -- reduces peak VRAM during decode
- **720P native** -- 1280x720 default resolution, 81 frames @ 16 fps (5 seconds)

## Supported Resolutions

| Resolution | Width | Height |
|------------|-------|--------|
| 720P landscape | 1280 | 720 |
| 720P portrait | 720 | 1280 |
| 480P landscape | 832 | 480 |
| 480P portrait | 480 | 832 |

## Frame Count

Wan2.2 requires `num_frames = 4k + 1`. Valid values: 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, **81** (default), 85, ...

The handler auto-rounds to the nearest valid value. At 16 fps, 81 frames = 5.0 seconds of video.

## Quick Start

### Build with models baked in

```bash
# Basic build (downloads ~60-130 GB model during build)
docker build -t sneaky-vidgen .

# With HuggingFace token (for gated models)
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t sneaky-vidgen .

# With LoRAs baked in
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg LORA_URLS="Remade-AI/Rotate/rotate_20_epochs.safetensors" \
  -t sneaky-vidgen .
```

### Build without models (use network volume)

```bash
docker build --build-arg BAKE_MODELS=false -t sneaky-vidgen .
```

Mount your RunPod network volume at `/models/`. On first run the handler downloads the model automatically if not found.

### Run locally

```bash
docker run --gpus all -p 8000:8000 sneaky-vidgen
```

## API

### Generate Video

```json
{
  "input": {
    "image": "<base64-encoded image>",
    "prompt": "A cat riding a surfboard on ocean waves, cinematic",
    "negative_prompt": "blurry, static, low quality",
    "width": 1280,
    "height": 720,
    "steps": 30,
    "cfg_scale": 3.5,
    "num_frames": 81,
    "fps": 16,
    "seed": 0,
    "output_format": "mp4",
    "flow_shift": null,
    "loras": [
      {"name": "rotate_20_epochs", "strength": 0.8},
      {"name": "flyingEffect", "strength": 1.0}
    ]
  }
}
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | *required* | Base64-encoded input image (with or without data URI prefix) |
| `prompt` | string | `""` | Text description of desired video |
| `negative_prompt` | string | *(built-in)* | What to avoid in generation |
| `width` | int | `1280` | Video width (auto-aligned to model requirements) |
| `height` | int | `720` | Video height (auto-aligned to model requirements) |
| `steps` | int | `30` | Inference steps. Higher = better quality, slower. Range: 1-50 |
| `cfg_scale` | float | `3.5` | Classifier-free guidance scale. Range: 1.0-20.0 |
| `num_frames` | int | `81` | Number of frames (auto-rounded to 4k+1). 81 = 5s @ 16fps |
| `fps` | int | `16` | Output video frame rate. Model native rate is 16 |
| `seed` | int | `0` | Random seed. 0 = random |
| `output_format` | string | `"mp4"` | Video format: `mp4` or `webm` |
| `flow_shift` | float | `null` | Override flow shift for this request. null = use server default |
| `loras` | array | `[]` | Array of LoRAs to activate (see below) |
| `lora_name` | string | `null` | *Legacy*: single LoRA name (use `loras` instead) |
| `lora_strength` | float | `1.0` | *Legacy*: single LoRA strength (use `loras` instead) |

### LoRA Format

The `loras` field accepts an array of objects or strings:

```json
// Full format with individual strengths
"loras": [
  {"name": "rotate_20_epochs", "strength": 0.8},
  {"name": "flyingEffect", "strength": 1.2}
]

// Shorthand (strength defaults to 1.0)
"loras": ["rotate_20_epochs", "flyingEffect"]
```

LoRA names correspond to filenames (without `.safetensors`) in the `/models/loras/` directory.

### Meta Actions

Check worker status without running generation:

```json
{"input": {"action": "health"}}
{"input": {"action": "list_loras"}}
```

### Response

```json
{
  "video": "<base64-encoded video>",
  "video_format": "mp4",
  "seed": 1234567890,
  "params": {
    "width": 1280,
    "height": 720,
    "steps": 30,
    "cfg": 3.5,
    "num_frames": 81,
    "fps": 16,
    "quantization": "bnb4",
    "flow_shift": 5.0,
    "loras": [{"name": "rotate_20_epochs", "strength": 0.8}]
  }
}
```

## Environment Variables

Configure at deploy time via RunPod template or `docker run -e`.

### Model

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/wan2.2` | Path to the Wan2.2 diffusers model |
| `LORA_DIR` | `/models/loras` | Directory containing `.safetensors` LoRA files |
| `HF_TOKEN` | | HuggingFace token (for runtime model download) |

### Quantization

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `QUANT_METHOD` | `bnb4` | `bnb4`, `bnb8`, `none` | `bnb4` = NF4 4-bit (~20 GB VRAM), `bnb8` = 8-bit (~40 GB), `none` = BF16 (~66 GB) |

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SAGE_ATTENTION` | `true` | INT8 quantised attention (~2x speedup, no quality loss) |
| `CACHE_THRESHOLD` | `0` | FirstBlockCache threshold. `0` = disabled (max quality). Try `0.05`-`0.1` for speed |
| `TORCH_COMPILE` | `false` | JIT-compile transformers. First request is slow, subsequent ~1.5-2x faster |
| `ENABLE_CPU_OFFLOAD` | `true` | Move model components to CPU when not in use. Disable on high-VRAM GPUs |
| `DEFAULT_FLOW_SHIFT` | `5.0` | Scheduler flow shift. `3.0` for 480P, `5.0` for 720P |

## GPU Requirements

| QUANT_METHOD | Min VRAM | Recommended GPU |
|--------------|----------|-----------------|
| `bnb4` | 24 GB | RTX 4090, A5000, L40 |
| `bnb8` | 40 GB | A6000, L40S, A100 40GB |
| `none` | 66 GB | A100 80GB, H100 |

## Baking LoRAs at Build Time

Pass LoRA URLs via the `LORA_URLS` build arg (comma-separated):

```bash
docker build \
  --build-arg LORA_URLS="\
Remade-AI/Rotate/rotate_20_epochs.safetensors,\
https://civitai.com/api/download/models/1523247?type=Model&format=SafeTensor" \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=civitai_token,env=CIVITAI_TOKEN \
  -t sneaky-vidgen .
```

### Supported LoRA sources

| Format | Example |
|--------|---------|
| HuggingFace | `Remade-AI/Rotate/rotate_20_epochs.safetensors` |
| Direct URL | `https://huggingface.co/.../resolve/main/lora.safetensors` |
| CivitAI | `https://civitai.com/api/download/models/1523247?type=Model&format=SafeTensor` |

CivitAI downloads require `CIVITAI_TOKEN` passed as a build secret.

## Recommended Presets

### Maximum Quality (default)

```json
{
  "width": 1280, "height": 720,
  "steps": 30, "cfg_scale": 3.5,
  "num_frames": 81, "fps": 16
}
```

Server: `CACHE_THRESHOLD=0`, `QUANT_METHOD=bnb8` or `none`

### Fast Preview

```json
{
  "width": 832, "height": 480,
  "steps": 4, "cfg_scale": 3.5,
  "num_frames": 17, "fps": 16
}
```

Server: `CACHE_THRESHOLD=0.1`

### Balanced

```json
{
  "width": 1280, "height": 720,
  "steps": 15, "cfg_scale": 3.5,
  "num_frames": 49, "fps": 16
}
```

Server: `CACHE_THRESHOLD=0.05`

## Project Structure

```
sneaky-vidgen/
  handler.py          Main RunPod serverless handler
  builder.py          Build-time model & LoRA downloader
  Dockerfile          Container definition
  requirements.txt    Python dependencies
  test_input.json     Sample request (fast preset for deployment verification)
  .runpod/
    hub.json          RunPod Hub template configuration
    tests.json        Deployment verification tests
  .gitignore
```

## Flow Shift Guide

The `flow_shift` parameter controls the timestep schedule for flow matching. Higher values allocate more steps to high-noise denoising (better for high-resolution).

| Resolution | Recommended flow_shift |
|------------|----------------------|
| 480P (832x480) | 3.0 |
| 720P (1280x720) | 5.0 |

Override per-request via the `flow_shift` field, or set the server default with `DEFAULT_FLOW_SHIFT`.
