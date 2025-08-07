#!/bin/bash

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

MODEL_NAME="${LLM_MODEL:-unsloth/Qwen3-8B-FP8}"
PORT_URL="${LLM_PORT:-http://localhost:8444/v1}"
PORT=$(echo "$PORT_URL" | awk -F':' '{print $3}' | awk -F'/' '{print $1}')
LOG_FILE="vllm_chat.log"

echo "Starting Chat model: $MODEL_NAME on port $PORT"
echo "→ Full URL: $PORT_URL"
echo "→ Log file: $LOG_FILE"

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization 0.5 \
  --trust-remote-code > "$LOG_FILE" 2>&1 &

echo "✅ Chat model server started in background. Logs: $LOG_FILE"
