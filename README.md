# Sneaky Vidgen

RunPod serverless worker for **Wan2.2 Image-to-Video** generation using HuggingFace Diffusers.

Wan2.2 I2V-A14B is a Mixture-of-Experts model (27B total / 14B active) with two transformer stages: a **HighNoise** expert handling 90% of denoising and a **LowNoise** expert refining the final 10%.

## Features

- **Wan2.2 I2V-A14B** via `WanImageToVideoPipeline` (no ComfyUI dependency)
- **GGUF quantization** (default) -- baked into Docker image for instant cold starts (~34 GB image)
- **BitsAndBytes** -- NF4 4-bit or 8-bit as alternative (requires network volume for full model)
- **Multiple LoRAs** -- stack any combination with per-LoRA strength control
- **SageAttention** -- INT8 quantised attention, ~2x speedup with no quality loss
- **FirstBlockCache** -- TeaCache-style block caching (opt-in for speed)
- **torch.compile** -- opt-in JIT compilation for additional throughput
- **VAE tiling + slicing** -- reduces peak VRAM during decode
- **720P native** -- 1280x720 default resolution, 81 frames @ 16 fps (5 seconds)

## Quick Start

### Build (default: GGUF Q4_K_M baked in)

```bash
# Default (~34 GB image, fits RunPod 30-min build limit)
docker build --secret id=hf_token,env=HF_TOKEN -t sneaky-vidgen .

# Higher quality GGUF (~38 GB image)
docker build --secret id=hf_token,env=HF_TOKEN \
  --build-arg GGUF_QUANT=Q6_K -t sneaky-vidgen .

# With LoRAs baked in
docker build --secret id=hf_token,env=HF_TOKEN \
  --build-arg LORA_URLS="Remade-AI/Rotate/rotate_20_epochs.safetensors" \
  -t sneaky-vidgen .
```

### Available GGUF Quant Levels

| `GGUF_QUANT` build arg | Per transformer | Both | Total image |
|-------------------------|-----------------|------|-------------|
| `Q4_K_S` | 8.2 GB | 16.3 GB | ~32 GB |
| **`Q4_K_M`** (default) | 9.0 GB | 18.0 GB | ~34 GB |
| `Q5_K_M` | 10.1 GB | 20.1 GB | ~36 GB |
| `Q6_K` | 11.2 GB | 22.4 GB | ~38 GB |
| `Q8_0` | 14.4 GB | 28.7 GB | ~45 GB |

GGUF files are from [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF).

### Alternative: Network Volume (for BitsAndBytes / full precision)

For BNB quantization or full BF16, the full model (~60-130 GB) is too large to bake. Use a network volume instead:

```bash
# Slim image (~5 GB), model downloads on first startup
docker build --secret id=hf_token,env=HF_TOKEN \
  --build-arg QUANT_METHOD=bnb4 -t sneaky-vidgen .
```

At runtime, set `MODEL_PATH` to your network volume mount (e.g. `/runpod-volume/models/wan2.2`). The handler auto-downloads the model if not found.

## Supported Resolutions

| Resolution | Width | Height |
|------------|-------|--------|
| 720P landscape | 1280 | 720 |
| 720P portrait | 720 | 1280 |
| 480P landscape | 832 | 480 |
| 480P portrait | 480 | 832 |

## Frame Count

Wan2.2 requires `num_frames = 4k + 1`. Valid values: 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, **81** (default), 85, ...

Auto-rounded to the nearest valid value. At 16 fps, 81 frames = 5.0 seconds of video.

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
| `width` | int | `1280` | Video width (auto-aligned) |
| `height` | int | `720` | Video height (auto-aligned) |
| `steps` | int | `30` | Inference steps. Higher = better quality, slower. Range: 1-50 |
| `cfg_scale` | float | `3.5` | Guidance scale. Range: 1.0-20.0 |
| `num_frames` | int | `81` | Number of frames (auto-rounded to 4k+1). 81 = 5s @ 16fps |
| `fps` | int | `16` | Output frame rate. Native model rate is 16 |
| `seed` | int | `0` | Random seed. 0 = random |
| `output_format` | string | `"mp4"` | `mp4` or `webm` |
| `flow_shift` | float | `null` | Per-request flow shift override. null = server default |
| `loras` | array | `[]` | LoRAs to activate (see below) |
| `lora_name` | string | `null` | *Legacy*: single LoRA name |
| `lora_strength` | float | `1.0` | *Legacy*: single LoRA strength |

### LoRA Format

```json
// Full format with individual strengths
"loras": [
  {"name": "rotate_20_epochs", "strength": 0.8},
  {"name": "flyingEffect", "strength": 1.2}
]

// Shorthand (strength defaults to 1.0)
"loras": ["rotate_20_epochs", "flyingEffect"]
```

Names correspond to filenames (without `.safetensors`) in `/models/loras/`.

### Meta Actions

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
    "width": 1280, "height": 720,
    "steps": 30, "cfg": 3.5,
    "num_frames": 81, "fps": 16,
    "quantization": "gguf",
    "flow_shift": 5.0,
    "loras": [{"name": "rotate_20_epochs", "strength": 0.8}]
  }
}
```

## Environment Variables

### Model

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/wan2.2` | Diffusers model (text encoder, VAE, configs) |
| `GGUF_DIR` | `/models/gguf` | Directory with `HighNoise/` and `LowNoise/` GGUF files |
| `LORA_DIR` | `/models/loras` | LoRA `.safetensors` files |
| `HF_TOKEN` | | HuggingFace token (for runtime model download) |

### Quantization

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `QUANT_METHOD` | `gguf` | `gguf`, `bnb4`, `bnb8`, `none` | See GPU table below |

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SAGE_ATTENTION` | `true` | ~2x attention speedup, no quality loss |
| `CACHE_THRESHOLD` | `0` | FirstBlockCache. `0` = disabled (max quality). `0.05`-`0.1` for speed |
| `TORCH_COMPILE` | `false` | JIT-compile transformers. Slow first request, faster subsequent |
| `ENABLE_CPU_OFFLOAD` | `true` | Offload to CPU when idle. Disable on high-VRAM GPUs |
| `DEFAULT_FLOW_SHIFT` | `5.0` | `3.0` for 480P, `5.0` for 720P |

## GPU Requirements

| QUANT_METHOD | Min VRAM | Recommended GPU |
|--------------|----------|-----------------|
| `gguf` (Q4_K_M) | 16 GB | RTX 4090, L40, A5000 |
| `gguf` (Q6_K) | 24 GB | RTX 4090, L40, A5000 |
| `bnb4` | 24 GB | RTX 4090, A5000, L40 |
| `bnb8` | 40 GB | A6000, L40S, A100 40GB |
| `none` | 66 GB | A100 80GB, H100 |

## Baking LoRAs

```bash
docker build \
  --build-arg LORA_URLS="\
Remade-AI/Rotate/rotate_20_epochs.safetensors,\
https://civitai.com/api/download/models/1523247?type=Model&format=SafeTensor" \
  --secret id=hf_token,env=HF_TOKEN \
  --secret id=civitai_token,env=CIVITAI_TOKEN \
  -t sneaky-vidgen .
```

| Source | Format |
|--------|--------|
| HuggingFace | `owner/repo/filename.safetensors` |
| Direct URL | `https://huggingface.co/.../resolve/main/lora.safetensors` |
| CivitAI | `https://civitai.com/api/download/models/...` (needs `CIVITAI_TOKEN`) |

## Recommended Presets

### Maximum Quality (default)

```json
{"width": 1280, "height": 720, "steps": 30, "cfg_scale": 3.5, "num_frames": 81, "fps": 16}
```

Server: `CACHE_THRESHOLD=0`

### Fast Preview

```json
{"width": 832, "height": 480, "steps": 4, "cfg_scale": 3.5, "num_frames": 17, "fps": 16}
```

Server: `CACHE_THRESHOLD=0.1`

### Balanced

```json
{"width": 1280, "height": 720, "steps": 15, "cfg_scale": 3.5, "num_frames": 49, "fps": 16}
```

Server: `CACHE_THRESHOLD=0.05`

## GGUF Compatibility Notes

| Feature | GGUF | BNB / None |
|---------|------|------------|
| SageAttention | Yes | Yes |
| FirstBlockCache | Yes | Yes |
| VAE tiling/slicing | Yes | Yes |
| LoRA (unfused adapters) | Yes | Yes |
| LoRA fusing (`fuse_lora()`) | **No** | Yes |
| torch.compile | **No** | Yes |

- **LoRA fusing** is not supported with GGUF (shape mismatch between quantized and LoRA tensors). Unfused PEFT adapters work fine and are what the handler uses.
- **torch.compile** produces silently wrong results with GGUF due to PyTorch tracing bugs with custom tensor subclasses. The handler auto-skips it when `QUANT_METHOD=gguf`.

## Flow Shift Guide

| Resolution | Recommended `flow_shift` |
|------------|--------------------------|
| 480P (832x480) | 3.0 |
| 720P (1280x720) | 5.0 |

Override per-request via `flow_shift` field, or set server default with `DEFAULT_FLOW_SHIFT`.

## Project Structure

```
sneaky-vidgen/
  handler.py          RunPod serverless handler
  builder.py          Build-time model & LoRA downloader
  Dockerfile          Container (bakes GGUF by default)
  requirements.txt    Python dependencies
  test_input.json     Fast preset for deployment verification
  .runpod/
    hub.json          RunPod Hub template
    tests.json        Deployment verification tests
```
