# Usage Guide

Comprehensive guide for installing and using Chimera-M.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Directory Setup](#directory-setup)
- [Model Requirements](#model-requirements)
- [Dataset Formats](#dataset-formats)
- [Running Training](#running-training)
- [Common Workflows](#common-workflows)

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any x86_64 | 8+ cores |
| RAM | 8 GB | 32+ GB |
| GPU | None (CPU fallback) | 16+ GB VRAM |
| Storage | 10 GB free | 100+ GB SSD |
| OS | Linux/macOS/Windows | Linux |

### Dependencies

Chimera-M requires:
- Python 3.8+
- PyTorch 2.0+
- transformers (for model loading)
- numpy, psutil (system utilities)
- lz4 (for SSD offloading, gears 4-5)

### Install Commands

**Basic installation (CPU-only):**
```bash
pip install torch transformers numpy psutil lz4
```

**With CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy psutil lz4
```

**With CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy psutil lz4
```

**Verify installation:**
```bash
python -c "
import torch
import transformers
import lz4.frame
import psutil
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Quick Start

### 1. Create Folder Structure

```bash
mkdir -p chimera-m/Models chimera-m/Datasets chimera-m/Output
cd chimera-m
```

Folder purposes:
- `Models/` - Place exactly one model here
- `Datasets/` - Place one or more dataset files here
- `Output/` - Auto-created for checkpoints and logs

### 2. Download a Model

**Option A: From HuggingFace**
```bash
# Using git-lfs (for models < 5GB)
git lfs install
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct Models/phi-3
```

**Option B: Using Python**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # or "meta-llama/Llama-2-7b", etc.
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("Models/my-model")
tokenizer.save_pretrained("Models/my-model")
```

**Option C: Manual download**
Download model files (config.json, pytorch_model.bin or model.safetensors) to `Models/`.

### 3. Prepare Dataset

**Create sample data:**
```bash
# Plain text format
cat > Datasets/sample.txt << 'EOF'
This is a sample sentence for training.
Machine learning is fascinating.
Another line of training data.
EOF

# Or JSONL format
cat > Datasets/sample.jsonl << 'EOF'
{"text": "This is a sample sentence for training."}
{"text": "Machine learning is fascinating."}
{"text": "Another line of training data."}
EOF
```

**Real dataset sources:**
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [The Pile](https://pile.eleuther.ai/)
- Common Crawl dumps
- Custom curated data

### 4. Run Training

```bash
python train.py
```

Default behavior:
- Runs 10 epochs
- Auto-detects hardware and selects gear
- Saves checkpoints to `Output/` every 500 steps

**Expected first-run output:**
```
======================================================================
CHIMERA-M: Pre-Flight Check
======================================================================

Hardware:
  CUDA: Yes (24.0GB VRAM)
  RAM: 32.0GB
  CPU: 16 cores

Model:
  Format: llama
  Params: ~3.50B

Dataset:
  Files: 1
  Size: 15.3MB
  Lines: ~50,000

Status: ✓ PASSED

[OK] Pre-flight passed. Starting training...

======================================================================
CHIMERA-M: Hardware Detection
======================================================================
CUDA Available: True
VRAM: 24.0 GB
RAM: 32.0 GB
...
```

---

## Directory Setup

### Models Directory

**Location:** `./Models/`

**Constraints:**
- Exactly one model per run
- Must be HuggingFace Transformers format
- Supported architectures: Llama, GPT-2, Qwen, Mistral, Phi

**Structure:**
```
Models/
└── llama-3b/
    ├── config.json
    ├── pytorch_model.bin (or model.safetensors)
    ├── tokenizer.json
    └── tokenizer_config.json
```

**Multiple files handling:**
If `Models/` contains multiple items, the first alphabetically is used with a warning.

### Datasets Directory

**Location:** `./Datasets/`

**Supported formats:**
| Extension | Format | Auto-detected |
|-----------|--------|---------------|
| .jsonl | JSON Lines | Yes |
| .json | JSON array | Yes |
| .txt | Plain text | Yes |
| .csv | CSV with text column | Yes |
| .parquet | Apache Parquet | Yes |

**Structure:**
```
Datasets/
├── train.jsonl
├── validation.jsonl
└── extra_data.txt
```

All files are concatenated for training.

### Output Directory

**Location:** `./Output/` (auto-created)

**Contents:**
```
Output/
├── checkpoint_epoch1.pt
├── checkpoint_epoch2.pt
├── checkpoint_epochN.pt
├── gear_transition_step500_pre.pt
├── gear_transition_step500_post.pt
└── training.log
```

**Checkpoint format:**
```python
{
    'epoch': int,
    'step': int,
    'gear': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'loss': float,
}
```

---

## Model Requirements

### Supported Architectures

| Architecture | Format Detection | Expected Dataset |
|--------------|------------------|------------------|
| Llama | config.json with `llama` in arch | Chat format |
| Llama-2 | Same as Llama | Chat format |
| Llama-3 | Same as Llama | Chat format |
| Mistral | Same as Llama | Chat format |
| GPT-2 | `gpt2` in model type | Text format |
| Qwen | `qwen` in model name | Messages format |
| Phi | `phi` in model name | Chat format |

### Model Size Limits

| Hardware | Max Model | Recommended Gear |
|----------|-----------|------------------|
| 8 GB VRAM | 1-2B | 2-3 |
| 16 GB VRAM | 3-4B | 2-3 |
| 24 GB VRAM | 7-8B | 3-4 |
| 48 GB VRAM | 13-16B | 2-4 |
| CPU + 64GB RAM | 3B | 4-5 |

---

## Dataset Formats

### Chat Format (Llama, Mistral, Phi)

**Required structure:**
```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"}
]}
```

**Auto-conversion from text:**
If plain text provided, system wraps:
```
[APPLYING FIX] Wrapping with system prompt
```

Result:
```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "<original text>"}
]}
```

### Text Format (GPT-2)

**Accepted formats:**
```jsonl
{"text": "The quick brown fox..."}
```

or plain text:
```
The quick brown fox...
Jumps over the lazy dog.
```

### Messages Format (Qwen)

**Structure:**
```jsonl
{
  "id": "identity_0",
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Hello!"},
    {"from": "gpt", "value": "Hi there!"}
  ]
}
```

---

## Running Training

### Basic Command

```bash
python train.py
```

Uses all defaults.

### Common Options

**Adjust training duration:**
```bash
python train.py --epochs 50
```

**Adjust learning:**
```bash
python train.py --lr 2e-4 --batch-size 2
```

**Limit sequence length (reduce memory):**
```bash
python train.py --max-length 256
```

**Force compression level:**
```bash
python train.py --gear 3
```

**Resume training:**
```bash
python train.py --resume ./Output/checkpoint_epoch5.pt
```

**Disable auto-tuning:**
```bash
python train.py --bo-off
```

**Adjust RAM threshold:**
```bash
python train.py --ssdr-threshold 0.90
```

### Full Command Reference

```bash
python train.py \
    --epochs 50 \
    --lr 2e-4 \
    --batch-size 2 \
    --max-length 512 \
    --log-interval 10 \
    --checkpoint-interval 500 \
    --gear auto \
    --ssdr-threshold 0.85 \
    --models-dir ./Models \
    --datasets-dir ./Datasets \
    --output-dir ./Output
```

---

## Common Workflows

### Workflow 1: Test on Small Model

Verify setup works:
```bash
# Download small model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.save_pretrained('Models/gpt2')
"

# Create tiny dataset
echo "Hello world" > Datasets/test.txt

# Train 1 epoch
python train.py --epochs 1 --max-length 128
```

### Workflow 2: Train 7B on 24GB GPU

```bash
# Place 7B model in Models/
python train.py \
    --epochs 10 \
    --batch-size 1 \
    --max-length 512 \
    --gear auto
```

Expected: Auto-selects gear 3, occasional shifts to 4.

### Workflow 3: CPU-Only Training

```bash
# Start with high compression
python train.py \
    --gear 5 \
    --epochs 5 \
    --max-length 256 \
    --batch-size 1
```

Note: Very slow. Consider cloud GPU for models > 250M.

### Workflow 4: Resume After Interruption

```bash
# List checkpoints
ls -t Output/checkpoint_*.pt | head -5

# Resume from latest
python train.py --resume Output/checkpoint_epoch3.pt
```

Gear level is preserved from checkpoint.

### Workflow 5: Fine-tune with Custom Data

```bash
# 1. Prepare dataset in correct format
python prepare_data.py --input raw_data.txt --output Datasets/formatted.jsonl

# 2. Verify format
python -c "
from chimera_m import infer_dataset_format, scan_datasets
files = scan_datasets('Datasets')
print(infer_dataset_format(files))
"

# 3. Train
python train.py --epochs 3 --lr 1e-4
```

### Workflow 6: Multi-Experiment Comparison

```bash
# Experiment 1: Gear 2
python train.py --gear 2 --output-dir ./Output/exp_gear2

# Experiment 2: Gear 3
python train.py --gear 3 --output-dir ./Output/exp_gear3

# Compare results
python compare.py ./Output/exp_gear2 ./Output/exp_gear3
```

---

## Monitoring Training

### Console Output

Normal training shows:
```
Epoch 0 [50/1000] Loss: 2.3456 Gear: 3 Step: 0.512s
Epoch 0 [60/1000] Loss: 2.2987 Gear: 3 Step: 0.508s
```

Gear shifts appear as:
```
[INFO] Gear shift: 3 → 4 at step 500
[INFO] Checkpoint: gear_transition_step500_pre.pt
[INFO] Checkpoint: gear_transition_step500_post.pt
```

### Log Files

Logs written to `Output/training.log` (if configured) or console.

### Check Progress

```bash
# Watch live
tail -f Output/training.log

# Check latest checkpoint
ls -lt Output/checkpoint_*.pt | head -1
```

---

## Stopping and Resuming

### Graceful Stop

Press `Ctrl+C` once:
- Saves checkpoint at current step
- Cleans up SSD cache
- Exits cleanly

### Emergency Stop

Press `Ctrl+C` twice:
- Immediate exit
- Checkpoint may be corrupted

### Resume

```bash
python train.py --resume Output/checkpoint_epochN.pt
```

Restores:
- Model weights
- Optimizer state
- Current gear level
- Training step/epoch

---

## Troubleshooting First Run

### "No module named 'lz4'"
```bash
pip install lz4
```

### "No model found"
```bash
ls Models/  # Should show model folder
cp -r /path/to/model Models/
```

### "CUDA out of memory" on start
```bash
# Start with higher compression
python train.py --gear 5 --max-length 128
```

### Very slow on first run
Normal - system is compiling CUDA kernels and detecting hardware.

### Preflight shows warnings
Warnings don't block training. Errors do. Fix errors, ignore warnings if acceptable.
