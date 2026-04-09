FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY builder.py handler.py ./

# Build args for model configuration
ARG BAKE_MODELS=true
ARG MODEL_ID=Wan-AI/Wan2.2-I2V-A14B-Diffusers
ARG LORA_URLS=

ENV MODEL_ID=${MODEL_ID}
ENV LORA_URLS=${LORA_URLS}

# Download models during build
# Pass tokens via BuildKit secrets, not ARG/ENV:
#   docker build --secret id=hf_token,env=HF_TOKEN --secret id=civitai_token,env=CIVITAI_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    --mount=type=secret,id=civitai_token \
    if [ "${BAKE_MODELS}" = "true" ]; then \
      export HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || echo "") && \
      export CIVITAI_TOKEN=$(cat /run/secrets/civitai_token 2>/dev/null || echo "") && \
      python3 builder.py; \
    fi

# Default runtime configuration
ENV MODEL_PATH=/models/wan2.2
ENV LORA_DIR=/models/loras
ENV QUANT_METHOD=bnb4
ENV ENABLE_SAGE_ATTENTION=true
ENV CACHE_THRESHOLD=0
ENV ENABLE_CPU_OFFLOAD=true
ENV DEFAULT_FLOW_SHIFT=5.0
ENV TORCH_COMPILE=false

CMD ["python3", "-u", "handler.py"]
