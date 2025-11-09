#!/bin/bash

# Real-Time Inference Client Launcher
# Usage: ./scripts/start_client.sh [stream_name] [stream_url]
#
# Examples:
#   ./scripts/start_client.sh                    # Default: cam_1, webcam
#   ./scripts/start_client.sh cam_2 0            # cam_2, webcam
#   ./scripts/start_client.sh cam_3 rtsp://...   # cam_3, RTSP stream

STREAM_NAME=${1:-"cam_1"}
STREAM_URL=${2:-"0"}

echo "=========================================="
echo "Starting Inference Client"
echo "Stream: $STREAM_NAME"
echo "Source: $STREAM_URL"
echo "=========================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p output/$STREAM_NAME

# Create logs directory
mkdir -p logs

# Export config as environment variables
export STREAM_NAME=$STREAM_NAME
export STREAM_URL=$STREAM_URL

# Start client
echo "Launching client..."
python -m src.client.client

# Note: Use Ctrl+C to stop the client
