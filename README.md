# Qwen3-VL-30B Fine-Tuning Setup

This directory contains configuration files and scripts for fine-tuning Qwen3-VL-30B models using Axolotl with DeepSpeed or Ray for distributed training.

## Quick Start

### Data pre-processing
Data should be arranged in one or more jsonl files with user-assistant conversations.

### 1. Environment Setup
```bash
bash setup.sh
```

### 2. Download Model (Optional)
```bash
python download_model.py
```

### 3. Launch Training

#### Option A: DeepSpeed (Single Node, Multi-GPU)

**Basic command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 axolotl train qwen3-vl-deepspeed.yml
```

**With accelerate launcher (recommended for better CPU threading):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
accelerate launch --config_file accelerate_configs/accelerate_config_single_node.yaml \
  -m axolotl.cli.train qwen3-vl-deepspeed.yml > training_log.txt 2>&1
```

#### Option B: Ray (Multi-Node Distributed)

**Set up cluster:**

1. Start head node (in tmux):
```bash
ray start --head
```

2. Copy the join command and run it on each worker node (in tmux)

3. Check cluster status:
```bash
ray status
```

**Run training job:**
```bash
RAY_TRAIN_V2_ENABLED=1 \
HF_HOME="/mnt/haonan-us-1b/hf" \
axolotl train qwen3-vl-multi-node.yml --use-ray
```

Note: Update `ray_num_workers` in `qwen3-vl-multi-node.yml` to match your cluster size.
