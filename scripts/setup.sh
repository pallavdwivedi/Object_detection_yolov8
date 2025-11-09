#!/bin/bash

# Setup script for Real-Time Inference System
# Usage: ./scripts/setup.sh

echo "=========================================="
echo "Setting up Real-Time Inference System"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p models
mkdir -p logs
mkdir -p output
mkdir -p config

# Download YOLOv8 model (will auto-download on first run, but we can pre-download)
echo "YOLOv8 model will be downloaded automatically on first run"

# Make scripts executable
chmod +x scripts/*.sh

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start server: ./scripts/start_server.sh"
echo "2. Start client: ./scripts/start_client.sh"
echo ""
echo "Or activate venv manually:"
echo "  source venv/bin/activate"
echo "  python -m src.server.server"
echo "  python -m src.client.client"
