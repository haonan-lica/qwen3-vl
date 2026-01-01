# Environment Setup Guide

This guide explains how to reproduce the `ray_vl_new` environment for Qwen3-VL MoE training.

## Prerequisites

- Conda (miniconda or anaconda)
- CUDA 13.0 compatible system
- Sufficient disk space (~10GB for environment)

## Method 1: Using setup.sh (Recommended)

The `setup.sh` script provides the most control and includes all necessary configuration steps.

### Basic Usage

```bash
# Create environment with default name (ray_vl_new)
bash setup.sh

# Or create with a custom name
bash setup.sh my_custom_env_name
```

### What the script does

1. Creates a conda environment with Python 3.12.12
2. Installs PyTorch 2.9.1 with CUDA 13.0 support
3. Installs all required dependencies from `requirements.txt`
4. Installs Flash Attention 2.8.3 (required for efficient attention)
5. Installs Axolotl v0.13.0 from GitHub (includes Qwen3-VL MoE support)
6. Installs DeepSpeed 0.18.2 for distributed training
7. Installs xformers 0.0.33.post2 for PyTorch 2.9 compatibility
8. Fixes the telemetry whitelist.yaml issue
9. Installs Ray for distributed training support

### Environment Protection

The script includes a check to prevent accidentally overwriting the `ray_vl_new` environment. If you try to create an environment that already exists, it will prompt you to either remove the existing one or use a different name.

## Method 2: Using environment.yml

Alternatively, you can use the conda environment file:

```bash
conda env create -f environment.yml
conda activate ray_vl_new_reproduced
```

Note: This method may take longer as conda needs to resolve all dependencies, and it may not include some of the custom configuration steps from setup.sh.

## Verification

After setup, verify your installation:

### 1. Activate the environment

```bash
conda activate ray_vl_new  # or your custom environment name
```

### 2. Check PyTorch and CUDA

```bash
python -c 'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA: {torch.version.cuda}"); print(f"CUDA available: {torch.cuda.is_available()}")'
```

Expected output:
```
PyTorch: 2.9.1+cu130
CUDA: 13.0
CUDA available: True
```

### 3. Verify Qwen3-VL MoE support

```bash
CUDA_VISIBLE_DEVICES='' python -c 'from transformers import AutoConfig; config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True); print(f"✅ Model type: {config.model_type}")'
```

### 4. Verify Flash Attention

```bash
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
CUDA12_LIB="${SITE_PACKAGES}/nvidia/cuda_runtime/lib" && \
CUDA13_LIB="${SITE_PACKAGES}/nvidia/cu13/lib" && \
LD_LIBRARY_PATH="${CUDA12_LIB}:${CUDA13_LIB}:${LD_LIBRARY_PATH}" \
CUDA_VISIBLE_DEVICES='' python -c 'from flash_attn import flash_attn_func; print("✅ Flash Attention working")'
```

## Key Package Versions

The environment includes the following key packages (matching `ray_vl_new`):

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu130 |
| Transformers | 4.57.1 |
| Axolotl | 0.13.0.dev0 |
| DeepSpeed | 0.18.2 |
| Flash Attention | 2.8.3 |
| xformers | 0.0.33.post2 |
| wandb | 0.23.1 |
| accelerate | 1.11.0 |
| peft | 0.18.0 |
| Ray | (latest with train support) |

## Important Notes

### CUDA Library Paths

When running training, ensure `LD_LIBRARY_PATH` includes both CUDA 12 and 13 runtime libraries:

- Flash Attention binary uses CUDA 12
- PyTorch 2.9.1+cu130 uses CUDA 13

This is already configured in training scripts like `train_with_local_cache.sh`.

### File Structure

```
qwen3-vl/
├── setup.sh                    # Main setup script (recommended)
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python package requirements
├── requirements_ray.txt        # Alternative requirements file (legacy)
└── ENVIRONMENT_SETUP.md        # This file
```

## Troubleshooting

### PyTorch CUDA mismatch

If you see CUDA version mismatches, ensure:
1. Your system has CUDA 13.0 installed
2. You installed PyTorch with the correct CUDA version: `torch==2.9.1+cu130`

### Flash Attention import errors

If Flash Attention fails to import:
1. Check that both CUDA 12 and 13 runtime libraries are in your `LD_LIBRARY_PATH`
2. Reinstall with: `pip install flash-attn==2.8.3 --no-build-isolation`

### Axolotl telemetry errors

The setup script automatically fixes the missing `whitelist.yaml` file. If you still encounter telemetry errors, manually download:

```bash
AXOLOTL_PATH=$(python3 -c "import axolotl; import os; print(os.path.dirname(axolotl.__file__))")
curl -sL https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/v0.13.0/src/axolotl/telemetry/whitelist.yaml \
  -o "$AXOLOTL_PATH/telemetry/whitelist.yaml"
```

## Updating the Environment

To update the environment specification after making changes to `ray_vl_new`:

1. Export the environment:
   ```bash
   conda env export -n ray_vl_new --from-history > requirements_snapshot.txt
   ```

2. Review and update `requirements.txt` with any new packages

3. Update version numbers in `setup.sh` if needed

4. Test the setup script with a new environment name to verify it works:
   ```bash
   bash setup.sh test_env
   conda env remove -n test_env
   ```
