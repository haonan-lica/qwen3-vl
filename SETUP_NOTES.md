# Qwen3-VL MoE Training Environment Setup

## Overview

This environment is specifically configured for training Qwen3-VL MoE (Mixture of Experts) models using Axolotl with Ray and DeepSpeed support.

## Quick Start

```bash
bash setup.sh
conda activate ray_vl_new
```

## Key Components

### Environment: `ray_vl_new`
- **Python:** 3.12
- **CUDA:** 13.0 (matching system CUDA to avoid compatibility issues)

### Core Packages
- **PyTorch:** 2.9.1+cu130
- **Transformers:** 4.57.1 (Qwen3-VL MoE support)
- **Axolotl:** 0.13.0.dev0 (from GitHub v0.13.0 release)
- **Flash Attention:** 2.8.3+
- **Ray:** 2.52.1+ (optional)
- **Accelerate:** 1.11.0+

## Why CUDA 13.0 and PyTorch 2.9?

**Important:** This setup uses CUDA 13.0 to match your system CUDA version, avoiding PyTorch CUDA vs system CUDA compatibility issues that can cause unexpected CUDA errors during training.

Axolotl v0.13.0 officially supports:
- PyTorch 2.8 and 2.9 (CUDA 13.0 is supported)
- CUDA 12.8 and 13.0

## Why Axolotl v0.13.0 from GitHub?

**Critical:** The PyPI version only has 0.13.0.dev0, not the stable 0.13.0 release.

Axolotl v0.13.0 (released December 2024) added:
- Full support for `qwen3_vl_moe` model architecture
- Updated transformers dependency to 4.57.1
- Enhanced multimodal training capabilities
- PyTorch 2.9 compatibility

## Installation Notes

### Build Tools Required
The setup requires build tools (gcc, g++, ninja) for compiling:
- Flash Attention kernels
- Custom CUDA extensions

### GPU Support
All packages are installed with CUDA 13.0 support but **do not** require GPU access during installation. The installation process uses `CUDA_VISIBLE_DEVICES=""` where needed to avoid interfering with running training jobs.

### Dependency Resolution

**IMPORTANT:** DeepSpeed must be installed separately, not via `axolotl[deepspeed]`.

Installing `axolotl[deepspeed]` would downgrade PyTorch from 2.9.1+cu130 to 2.8.0, losing CUDA 13.0 support.

Installation order:
1. PyTorch 2.9.1 with CUDA 13.0
2. Axolotl v0.13.0 (without extras)
3. DeepSpeed 0.18.2 (separately)
4. xformers upgrade (for PyTorch 2.9 compatibility)

Axolotl v0.13.0 installs the correct versions of:
- transformers (4.57.1)
- accelerate (1.11.0)
- datasets (4.4.1)
- peft (0.18.0)
- trl (0.25.0)

This ensures full compatibility with Qwen3-VL MoE models.

## Verification

After installation, verify Qwen3-VL MoE support:

```bash
CUDA_VISIBLE_DEVICES='' python -c '
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True)
print(f"✅ Model type: {config.model_type}")
print(f"✅ Architecture: {config.architectures}")
'
```

Expected output:
```
✅ Model type: qwen3_vl_moe
✅ Architecture: ['Qwen3VLMoeForConditionalGeneration']
```

## Training Configuration

Your existing config files should work with minor adjustments:
- `qwen3-vl-deepspeed.yml` - DeepSpeed training config
- `qwen3-vl-multi-node.yml` - Multi-node Ray training config

### Key Config Parameters for Qwen3-VL MoE:
```yaml
base_model: Qwen/Qwen3-VL-30B-A3B-Instruct
processor_type: AutoProcessor
load_in_4bit: true
adapter: qlora

# Vision + Language model settings
skip_prepare_dataset: false
remove_unused_columns: true
```

## Troubleshooting

### Issue: `qwen3_vl_moe` not recognized
**Solution:** Ensure transformers >= 4.57.0 is installed. Run:
```bash
pip show transformers
```

### Issue: Flash Attention compilation fails
**Solution:** Ensure build tools are installed:
```bash
conda install -c conda-forge gcc gxx
```

### Issue: CUDA version mismatch
**Solution:** This environment uses CUDA 13.0 to match system CUDA, avoiding PyTorch CUDA vs system CUDA compatibility issues. Ensure your system has CUDA 13.0 drivers.

### Issue: Flash Attention ImportError `libcudart.so.12: cannot open shared object file`
**Solution:** Flash attention was compiled against CUDA 12 while PyTorch uses CUDA 13. Both runtime libraries need to be in LD_LIBRARY_PATH:
```bash
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
CUDA12_LIB_PATH="${SITE_PACKAGES}/nvidia/cuda_runtime/lib"
CUDA13_LIB_PATH="${SITE_PACKAGES}/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${CUDA12_LIB_PATH}:${CUDA13_LIB_PATH}:${LD_LIBRARY_PATH}"
```
This is already configured in `/mnt/haonan-us-1b/projects/qwen-text2svg-axolotl/train_with_local_cache.sh`.

## Differences from Previous Environments

| Component | Previous | ray_vl_new | Reason |
|-----------|----------|------------|---------|
| Python | 3.11 | 3.12 | Better compatibility |
| PyTorch | 2.6.0/2.8.0 | 2.9.1+cu130 | CUDA 13.0 system compatibility |
| CUDA | 12.4/12.8 | 13.0 | Match system CUDA, avoid errors |
| Axolotl | 0.12.x | 0.13.0.dev0 (GitHub v0.13.0) | Qwen3-VL MoE support |
| Transformers | 4.55.x | 4.57.1 | `qwen3_vl_moe` architecture |

## Resources

- [Axolotl v0.13.0 Release](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.13.0)
- [Axolotl Documentation](https://docs.axolotl.ai/)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- [Qwen Training Guide](https://qwen.readthedocs.io/en/latest/training/axolotl.html)
