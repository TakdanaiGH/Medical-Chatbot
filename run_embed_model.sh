#!/bin/bash

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

MODEL_NAME="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
PORT_URL="${EMBEDDING_PORT:-http://localhost:8555/v1}"
PORT=$(echo "$PORT_URL" | awk -F':' '{print $3}' | awk -F'/' '{print $1}')
LOG_FILE="vllm_embed.log"

echo "Starting Embedding model: $MODEL_NAME on port $PORT"
echo "→ Full URL: $PORT_URL"
echo "→ Log file: $LOG_FILE"

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --task embed \
  --port "$PORT" \
  --gpu-memory-utilization 0.25 \
  --trust-remote-code > "$LOG_FILE" 2>&1 &

echo "✅ Embedding model server started in background. Logs: $LOG_FILE"
