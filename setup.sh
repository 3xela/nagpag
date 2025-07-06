#!/bin/bash

set -e

echo "Setting up virtual environment with uv..."
uv venv
source .venv/bin/activate
uv pip install fal

echo "Setup complete! Virtual environment created and fal installed."