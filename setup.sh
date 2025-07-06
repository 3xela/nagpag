#!/bin/bash

set -e

echo "Setting up virtual environment with uv..."
uv venv
source .venv/bin/activate
uv pip install fal

echo "Setup complete! Virtual environment created and fal installed."

echo "Checking fal authentication..."
auth_output=$(fal auth whoami 2>&1)
if [[ "$auth_output" != Hello* ]]; then
    echo "Not authenticated. Running fal auth login..."
    fal auth login
else
    echo "Already authenticated: $auth_output"
fi