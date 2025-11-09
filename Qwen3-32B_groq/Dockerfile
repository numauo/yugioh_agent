FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir requests

# Copy duel agent sources.
COPY Qwen3-32B_groq.py ./Qwen3-32B_groq.py

# Repository root (prompt/decks) is expected to be mounted at runtime.
# GPU 実行例:
#   docker run --rm -it --gpus all \
#     -e GROQ_API_KEY=sk-... \
#     -v "$PWD":/workspace \
#     groq-ygo \
#     python3 Qwen3-32B_groq.py \
#       --prompt-file /workspace/prompt_yugioh.txt \
#       --user-deck-json /workspace/user-deck.json \
#       --ai-deck-json /workspace/ai-deck.json

ENTRYPOINT ["python3", "Qwen3-32B_groq.py"]
