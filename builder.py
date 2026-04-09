"""
Build-time model downloader for Sneaky Vidgen.

Two modes controlled by QUANT_METHOD:

  gguf (default) -- Downloads lightweight diffusers model (text encoder, VAE,
      configs only -- no transformer weights) plus GGUF-quantised transformer
      files from QuantStack. Total ~28-38 GB depending on quant level.

  bnb4 / bnb8 / none -- Downloads the full diffusers model (~60-130 GB).
      Too large to bake into a Docker image; use a network volume instead.

Environment variables:
  QUANT_METHOD   gguf (default), bnb4, bnb8, none
  GGUF_QUANT     Q4_K_M (default), Q4_K_S, Q5_K_M, Q6_K, Q8_0
  MODEL_ID       Wan-AI/Wan2.2-I2V-A14B-Diffusers
  MODEL_PATH     /models/wan2.2
  GGUF_REPO      QuantStack/Wan2.2-I2V-A14B-GGUF
  GGUF_DIR       /models/gguf
  LORA_URLS      Comma-separated HF paths or URLs
  LORA_DIR       /models/loras
  HF_TOKEN       HuggingFace token (optional)
  CIVITAI_TOKEN  CivitAI token (optional)
"""

import os
import re
import requests

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/wan2.2")
GGUF_REPO = os.environ.get("GGUF_REPO", "QuantStack/Wan2.2-I2V-A14B-GGUF")
GGUF_DIR = os.environ.get("GGUF_DIR", "/models/gguf")
GGUF_QUANT = os.environ.get("GGUF_QUANT", "Q4_K_M")
QUANT_METHOD = os.environ.get("QUANT_METHOD", "gguf")
LORA_DIR = os.environ.get("LORA_DIR", "/models/loras")
LORA_URLS = os.environ.get("LORA_URLS", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")


def download_diffusers_model(skip_transformers=False):
    """Download the Wan2.2 diffusers model from HuggingFace."""
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping.")
        return

    ignore = None
    if skip_transformers:
        ignore = [
            "transformer/diffusion_pytorch_model*",
            "transformer_2/diffusion_pytorch_model*",
        ]
        print(f"Downloading {MODEL_ID} (skipping transformer weights) ...")
    else:
        print(f"Downloading {MODEL_ID} (full model) ...")

    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_PATH,
        ignore_patterns=ignore,
        token=HF_TOKEN or None,
    )
    print(f"Saved to {MODEL_PATH}")


def download_gguf_files():
    """Download GGUF-quantised transformer files from QuantStack."""
    from huggingface_hub import hf_hub_download

    os.makedirs(GGUF_DIR, exist_ok=True)

    files = [
        f"HighNoise/Wan2.2-I2V-A14B-HighNoise-{GGUF_QUANT}.gguf",
        f"LowNoise/Wan2.2-I2V-A14B-LowNoise-{GGUF_QUANT}.gguf",
    ]

    for filename in files:
        local = os.path.join(GGUF_DIR, filename)
        if os.path.exists(local):
            print(f"  Already exists: {filename}")
            continue

        print(f"  Downloading {GGUF_REPO}/{filename} ...")
        hf_hub_download(
            GGUF_REPO,
            filename=filename,
            local_dir=GGUF_DIR,
            token=HF_TOKEN or None,
        )

    print(f"GGUF files saved to {GGUF_DIR}")


# ---------------------------------------------------------------------------
# LoRA downloads (same as before)
# ---------------------------------------------------------------------------
def download_file(url, output_dir, token=None):
    os.makedirs(output_dir, exist_ok=True)

    if "civitai.com" in url and token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={token}"

    print(f"  Downloading: {url}")
    try:
        resp = requests.get(url, stream=True, allow_redirects=True)
        resp.raise_for_status()

        cd = resp.headers.get("content-disposition", "")
        match = re.findall(r"filename[^;=\n]*=((['\"]).*?\2|[^;\n]*)", cd)
        filename = match[0][0].strip("'\"") if match and match[0][0] else url.split("/")[-1].split("?")[0]
        if not filename.endswith(".safetensors"):
            filename = "model.safetensors"

        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            print(f"  Already exists: {filename}")
            return

        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and done % (100 * 1024 * 1024) < 8192:
                        print(f"    {done // (1024*1024)} / {total // (1024*1024)} MB")

        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Error: {e}")


def download_hf_lora(source, output_dir):
    from huggingface_hub import hf_hub_download

    parts = source.split("/")
    if len(parts) < 3:
        print(f"  Invalid HF LoRA: {source} (expected owner/repo/file.safetensors)")
        return

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:])
    dest = os.path.join(output_dir, parts[-1])

    if os.path.exists(dest):
        print(f"  Already exists: {parts[-1]}")
        return

    print(f"  Downloading {repo_id}/{filename} ...")
    hf_hub_download(repo_id, filename=filename, local_dir=output_dir, token=HF_TOKEN or None)

    expected = os.path.join(output_dir, filename)
    if expected != dest and os.path.exists(expected):
        os.rename(expected, dest)


def download_loras():
    if not LORA_URLS:
        return
    os.makedirs(LORA_DIR, exist_ok=True)
    print("Downloading LoRAs ...")
    for entry in LORA_URLS.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry.startswith("http"):
            tok = CIVITAI_TOKEN if "civitai.com" in entry else None
            download_file(entry, LORA_DIR, token=tok)
        else:
            download_hf_lora(entry, LORA_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"Sneaky Vidgen Builder  (QUANT_METHOD={QUANT_METHOD})")
    print("=" * 60)

    if QUANT_METHOD == "gguf":
        print(f"GGUF mode: quant={GGUF_QUANT}")
        download_diffusers_model(skip_transformers=True)
        download_gguf_files()
    else:
        print(f"Full model mode ({QUANT_METHOD})")
        download_diffusers_model(skip_transformers=False)

    download_loras()

    print("=" * 60)
    print("Build complete!")
    print("=" * 60)
