#!/bin/bash

echo "🛠️  SETTING UP AXOLOTL+Ray TRAINING ENVIRONMENT"

# Create conda environment with Python 3.11
echo "Creating ray conda environment with Python 3.11..."
conda create -n ray_vl python=3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ray_vl

# Install required build tools first
echo "Installing build dependencies..."
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja

pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu130
pip3 install -r requirements_ray.txt

pip install flash_attn==2.7.4.post1 --no-build-isolation

