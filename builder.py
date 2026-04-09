"""
Build-time model downloader for Sneaky Vidgen.

Downloads the Wan2.2 I2V model and optional LoRAs from HuggingFace
so they are baked into the Docker image for instant cold starts.

Configure via environment variables:
  MODEL_ID        - HuggingFace model ID (default: Wan-AI/Wan2.2-I2V-A14B-Diffusers)
  MODEL_PATH      - Local path to save the model (default: /models/wan2.2)
  LORA_URLS       - Comma-separated list of LoRA sources to download
                    Formats:
                      HuggingFace:  repo_id/filename.safetensors
                      Direct URL:   https://example.com/lora.safetensors
  LORA_DIR        - Local path for LoRA files (default: /models/loras)
  HF_TOKEN        - HuggingFace access token (optional, for gated models)
  CIVITAI_TOKEN   - CivitAI API token (optional, for CivitAI downloads)
"""

import os
import re
import requests
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/wan2.2")
LORA_DIR = os.environ.get("LORA_DIR", "/models/loras")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")

# Default LoRAs to bake in (comma-separated)
# Format: "repo_id/filename.safetensors" for HuggingFace
#         "https://..." for direct URLs
LORA_URLS = os.environ.get("LORA_URLS", "").strip()


def download_model():
    """Download the Wan2.2 diffusers model from HuggingFace."""
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping download.")
        return

    print(f"Downloading {MODEL_ID} to {MODEL_PATH}...")
    print("This will take a while for the full model (~60-130 GB).")

    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_PATH,
        token=HF_TOKEN or None,
    )
    print(f"Model downloaded to {MODEL_PATH}")


def download_file(url, output_dir, token=None):
    """Download a file from a URL with streaming and progress."""
    os.makedirs(output_dir, exist_ok=True)

    headers = {}
    if "civitai.com" in url and token:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}token={token}"

    print(f"Downloading: {url}")
    try:
        response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Extract filename from Content-Disposition or URL
        content_disp = response.headers.get("content-disposition", "")
        filename_match = re.findall(
            r"filename[^;=\n]*=((['\"]).*?\2|[^;\n]*)", content_disp
        )
        if filename_match and filename_match[0][0]:
            filename = filename_match[0][0].strip("'\"")
        else:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".safetensors"):
                filename = "model.safetensors"

        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            print(f"  Already exists: {filename}, skipping.")
            return output_path

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 100) < 8192:
                        pct = downloaded / total_size * 100
                        print(f"  {downloaded / 1024 / 1024:.0f} MB / {total_size / 1024 / 1024:.0f} MB ({pct:.0f}%)")

        print(f"  Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return None


def download_hf_lora(source, output_dir):
    """Download a LoRA from HuggingFace (format: repo_id/filename.safetensors)."""
    from huggingface_hub import hf_hub_download

    # Parse "owner/repo/path/to/file.safetensors"
    parts = source.split("/")
    if len(parts) < 3:
        print(f"  Invalid HF LoRA source: {source}")
        print(f"  Expected format: owner/repo/filename.safetensors")
        return None

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:])

    output_filename = parts[-1]
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"  Already exists: {output_filename}, skipping.")
        return output_path

    print(f"  Downloading from HF: {repo_id} / {filename}")
    try:
        downloaded = hf_hub_download(
            repo_id,
            filename=filename,
            local_dir=output_dir,
            token=HF_TOKEN or None,
        )
        # hf_hub_download may place in subdirectory; move to flat dir
        expected = os.path.join(output_dir, filename)
        if expected != output_path and os.path.exists(expected):
            os.rename(expected, output_path)
        print(f"  Saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"  Error downloading HF LoRA: {e}")
        return None


def download_loras():
    """Download configured LoRA files."""
    if not LORA_URLS:
        print("No LoRA URLs configured.")
        return

    os.makedirs(LORA_DIR, exist_ok=True)
    print(f"Downloading LoRAs to {LORA_DIR}...")

    for entry in LORA_URLS.split(","):
        entry = entry.strip()
        if not entry:
            continue

        if entry.startswith("http://") or entry.startswith("https://"):
            # Direct URL download
            token = CIVITAI_TOKEN if "civitai.com" in entry else None
            download_file(entry, LORA_DIR, token=token)
        else:
            # HuggingFace format: repo_id/filename.safetensors
            download_hf_lora(entry, LORA_DIR)


if __name__ == "__main__":
    print("=" * 60)
    print("Sneaky Vidgen - Build-time Model Download")
    print("=" * 60)

    download_model()
    download_loras()

    print("=" * 60)
    print("Build-time download complete!")
    print("=" * 60)
