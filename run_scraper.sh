#!/bin/bash

# Ensure the script exits on error
set -e

# Name of the Python script
SCRIPT_NAME="auto_scraper.py"

# Log file location
LOG_FILE="scraper.log"

# Activate virtual environment if needed
# source venv/bin/activate

# Run the script in background and log output
nohup python3 -u auto_scraper.py > scraper.log 2>&1 &
echo "✅ Scraper started in background. Logging to $LOG_FILE"
