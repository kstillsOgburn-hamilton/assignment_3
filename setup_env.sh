#!/bin/bash

# Setup script for Assignment 3 - Bi-LSTM on NVIDIA server
echo "Setting up environment for Assignment 3..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install compatible versions of PyTorch and related packages
echo "Installing PyTorch and dependencies..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install Lightning and related packages
echo "Installing PyTorch Lightning..."
pip install lightning==2.1.0

# Install other required packages
echo "Installing other dependencies..."
pip install transformers==4.35.0
pip install torchmetrics==1.2.0
pip install torchtext==0.16.0
pip install python-dotenv
pip install wandb

echo "✅ Environment setup complete!"
echo ""
echo "To run training, execute:"
echo "  cd src && python3 train.py"
