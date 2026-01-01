#!/bin/bash

# Setup script to reproduce the ray_vl_new environment
# This script creates a conda environment matching the ray_vl_new configuration
# with all required packages for Qwen3-VL MoE training

echo "🛠️  SETTING UP QWEN3-VL MOE TRAINING ENVIRONMENT WITH CUDA 13.0"

# Parse command line arguments
ENV_NAME="${1:-ray_vl_new}"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Environment '${ENV_NAME}' already exists!"
    echo "   To create a new environment, either:"
    echo "   1. Remove the existing one: conda env remove -n ${ENV_NAME}"
    echo "   2. Use a different name: ./setup.sh my_new_env_name"
    exit 1
fi

# Create new conda environment with Python 3.12.12 (matching ray_vl_new)
echo "Creating ${ENV_NAME} conda environment with Python 3.12..."
conda create -n ${ENV_NAME} python=3.12.12 -y

# Activate the new environment
echo "Activating ${ENV_NAME} environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install required build tools first
echo "Installing build dependencies..."
pip3 install -U packaging setuptools wheel ninja

# Install PyTorch 2.9 with CUDA 13.0 support for system CUDA compatibility
# CUDA 13.0 matches your system CUDA version to avoid compatibility issues
echo "Installing PyTorch 2.9.1 with CUDA 13.0 support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install additional requirements (base dependencies)
echo "Installing additional dependencies..."
pip3 install -r requirements.txt

# Install Flash Attention (v2.8.3 - must be installed before Axolotl)
echo "Installing Flash Attention 2.8.3..."
pip3 install 'flash-attn==2.8.3' --no-build-isolation

# Install Axolotl v0.13.0 from GitHub (WITHOUT [deepspeed] extra to avoid PyTorch downgrade)
# This includes transformers 4.57.1+ which has Qwen3-VL MoE support
echo "Installing Axolotl v0.13.0 from GitHub (includes Qwen3-VL MoE support)..."
echo "Note: This may take several minutes and will install many dependencies..."
pip3 install 'git+https://github.com/axolotl-ai-cloud/axolotl.git@v0.13.0'

# Install DeepSpeed 0.18.2 separately (axolotl[deepspeed] would downgrade PyTorch to 2.8.0)
echo "Installing DeepSpeed 0.18.2 for distributed training..."
pip3 install 'deepspeed==0.18.2'

# Install xformers v0.0.33.post2 to be compatible with PyTorch 2.9
echo "Installing xformers 0.0.33.post2 for PyTorch 2.9 compatibility..."
pip3 install 'xformers==0.0.33.post2'

# Fix missing telemetry whitelist.yaml file (known issue in v0.13.0)
echo "Fixing missing telemetry whitelist.yaml..."
AXOLOTL_PATH=$(python3 -c "import axolotl; import os; print(os.path.dirname(axolotl.__file__))")
if [ ! -f "$AXOLOTL_PATH/telemetry/whitelist.yaml" ]; then
    curl -sL https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/v0.13.0/src/axolotl/telemetry/whitelist.yaml \
      -o "$AXOLOTL_PATH/telemetry/whitelist.yaml"
fi

# Install Ray for distributed training (optional)
echo "Installing Ray with training support..."
pip3 install 'ray[train]'

echo ""
echo "✅ Setup complete! Activate the environment with:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "Expected package versions (matching ray_vl_new):"
echo "  - Python: 3.12.12"
echo "  - PyTorch: 2.9.1+cu130"
echo "  - CUDA: 13.0 (matching system CUDA)"
echo "  - Transformers: 4.57.1"
echo "  - Axolotl: 0.13.0.dev0"
echo "  - DeepSpeed: 0.18.2"
echo "  - Flash Attention: 2.8.3"
echo "  - xformers: 0.0.33.post2"
echo "  - wandb: 0.23.1"
echo "  - accelerate: 1.11.0"
echo "  - peft: 0.18.0"
echo ""
echo "To verify installation, run:"
echo "   conda activate ${ENV_NAME}"
echo "   python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.version.cuda}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "To verify Qwen3-VL MoE support, run:"
echo "   CUDA_VISIBLE_DEVICES='' python -c 'from transformers import AutoConfig; config = AutoConfig.from_pretrained(\"Qwen/Qwen3-VL-30B-A3B-Instruct\", trust_remote_code=True); print(f\"✅ Model type: {config.model_type}\")'"
echo ""
echo "To verify Flash Attention, run:"
echo "   SITE_PACKAGES=\"\$(python3 -c 'import site; print(site.getsitepackages()[0])')\" && CUDA12_LIB=\"\${SITE_PACKAGES}/nvidia/cuda_runtime/lib\" && CUDA13_LIB=\"\${SITE_PACKAGES}/nvidia/cu13/lib\" && LD_LIBRARY_PATH=\"\${CUDA12_LIB}:\${CUDA13_LIB}:\${LD_LIBRARY_PATH}\" CUDA_VISIBLE_DEVICES='' python -c 'from flash_attn import flash_attn_func; print(\"✅ Flash Attention working\")'"
echo ""
echo "IMPORTANT: When running training, ensure LD_LIBRARY_PATH includes both CUDA 12 and 13 runtime libraries."
echo "Flash attention binary uses CUDA 12, while PyTorch 2.9.1+cu130 uses CUDA 13."
echo "This is already configured in train_with_local_cache.sh for qwen-text2svg-axolotl project."

