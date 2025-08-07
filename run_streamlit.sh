#!/bin/bash

# Load .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

ADDRESS="${STREAMLIT_ADDRESS:-localhost}"
PORT="${STREAMLIT_PORT:-8501}"
LOGFILE="streamlit_$(date +'%Y%m%d_%H%M%S').log"

echo "Starting Streamlit app at http://$ADDRESS:$PORT"
echo "→ Log file: $LOGFILE"

nohup streamlit run streamlit_app.py --server.address "$ADDRESS" --server.port "$PORT" > "$LOGFILE" 2>&1 &

echo "✅ Streamlit app running. Logs: $LOGFILE"
