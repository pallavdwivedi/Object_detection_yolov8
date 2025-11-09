#!/bin/bash

# Real-Time Inference Server Launcher
# Usage: ./scripts/start_server.sh

echo "=========================================="
echo "Starting Real-Time Inference Server"
echo "=========================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/yolov8n.pt" ]; then
    echo "Model not found. Downloading yolov8n.pt..."
    mkdir -p models
    # Model will be auto-downloaded by ultralytics on first run
fi

# Create logs directory
mkdir -p logs

# Create models directory
mkdir -p models

# Start server
echo "Launching server..."
python -m src.server.server

# Note: Use Ctrl+C to stop the server
