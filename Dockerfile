FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (ffmpeg for video encoding, git for HF downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY builder.py handler.py ./

# GGUF quant level: Q4_K_M (~34 GB image), Q6_K (~38 GB), Q8_0 (~45 GB)
ARG GGUF_QUANT=Q4_K_M
ARG LORA_URLS=

ENV QUANT_METHOD=gguf
ENV GGUF_QUANT=${GGUF_QUANT}
ENV LORA_URLS=${LORA_URLS}
ENV MODEL_PATH=/models/wan2.2
ENV GGUF_DIR=/models/gguf
ENV LORA_DIR=/models/loras
ENV ENABLE_SAGE_ATTENTION=true
ENV CACHE_THRESHOLD=0
ENV ENABLE_CPU_OFFLOAD=true
ENV DEFAULT_FLOW_SHIFT=5.0
ENV TORCH_COMPILE=false

# Download GGUF transformers + lightweight diffusers model at build time
# Pass HF_TOKEN via secret: docker build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=civitai_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || echo "") && \
    export CIVITAI_TOKEN=$(cat /run/secrets/civitai_token 2>/dev/null || echo "") && \
    python3 builder.py

CMD ["python3", "-u", "handler.py"]
