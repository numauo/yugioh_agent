# CUDA 12.1 + cuDNN8 ランタイム（軽量・実行専用）
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---- 環境変数 ----
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8 \
    LC_ALL=ja_JP.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    MODEL_ID=Qwen/Qwen3-32B

# ---- system deps & Python 3.12 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl gnupg tzdata git git-lfs \
    locales build-essential pkg-config ca-certificates \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
 && ln -sf /usr/bin/python3.12 /usr/bin/python && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python /tmp/get-pip.py && rm /tmp/get-pip.py \
 && echo "ja_JP.UTF-8 UTF-8" > /etc/locale.gen \
 && locale-gen && update-locale LANG=ja_JP.UTF-8 \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && git lfs install --system \
 && rm -rf /var/lib/apt/lists/*

# ---- Python tooling ----
RUN python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch (CUDA 12.1 対応ビルド) ----
RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# ---- HF/Transformers 周辺 ----
RUN python -m pip install --no-cache-dir \
    "transformers>=4.44.0" \
    "accelerate>=0.34.0" \
    "huggingface_hub>=0.25.0" \
    "safetensors>=0.4.0" \
    einops sentencepiece hf-transfer

# ※ 量子化は使わない前提（複数GPUでフル精度運用）。必要になったら bitsandbytes を追加。

# ---- キャッシュ ----
RUN mkdir -p /hf-cache && chmod -R 777 /hf-cache

# ---- アプリ配置 ----
WORKDIR /app
# ビルド時に同ディレクトリのファイルをコピー（マウント運用なら後で上書き可）
COPY yugioh_Qwen3_32B.py /app/yugioh_Qwen3_32B.py
COPY prompt_yugioh.txt /app/prompt_yugioh.txt

# 既定 CMD（MODEL_ID は ENV で Qwen3-32B-Instruct）
CMD ["python", "/app/yugioh_Qwen3_32B.py", "--prompt-file", "/app/prompt_yugioh.txt"]
