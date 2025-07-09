#!/bin/bash

# Setup script for meshai-schedule-extract using uv

echo "Setting up meshai-schedule-extract with uv..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the virtual environment, run: source .venv/bin/activate"
echo "To install additional packages, use: uv add <package-name>"