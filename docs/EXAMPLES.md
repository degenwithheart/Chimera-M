# Usage Examples

Complete examples for common scenarios.

## Basic Usage

### Simple Training

```bash
# Setup folders
mkdir -p Models Datasets Output

# Place model and data
cp -r /path/to/gpt2 Models/
cp /path/to/data.jsonl Datasets/

# Train with defaults
python train.py
```

### Custom Parameters

```bash
python train.py \
    --epochs 50 \
    --lr 2e-4 \
    --batch-size 2 \
    --max-length 1024
```

## Gear Selection

### Auto Gear (Recommended)

```bash
python train.py --gear auto
```

System automatically selects optimal gear based on hardware.

### Manual Gear

```bash
# Force Level 3 (20× compression)
python train.py --gear 3
```

Use when you know the constraints.

### Conservative (High Compression)

```bash
python train.py --gear 5 --epochs 100
```

For CPU-only or extreme memory constraints.

## Hardware Scenarios

### Large Model on Limited VRAM

**Scenario:** 8B model on 24GB GPU

```bash
python train.py \
    --gear auto \
    --batch-size 1 \
    --max-length 512
```

Expected: Auto-selects Level 3, occasional shifts to Level 4

### CPU-Only Training

**Scenario:** No CUDA available

```bash
python train.py \
    --gear 4 \
    --epochs 10 \
    --max-length 256
```

Expected: Level 4 (SSD offload), slower but works

### Multi-Epoch with Resume

```bash
# First run
python train.py --epochs 10

# Continue
python train.py --resume ./Output/checkpoint_epoch5.pt --epochs 20
```

## Advanced: Programmatic API

### Custom Training Loop

```python
from chimera_m import (
    ChimeraGearOptimizer,
    preflight_check,
    print_preflight_report,
    scan_datasets,
    GEAR_LEVELS
)
from pathlib import Path
import torch

# Setup
model_path = "./Models/llama-3b"
dataset_files = scan_datasets("./Datasets")

# Preflight
class Args:
    epochs = 10
    lr = 3e-4
    output_dir = "./Output"
    resume = None

args = Args()
results = preflight_check(model_path, dataset_files, 'auto', args)
print_preflight_report(results)

if not results['passed']:
    raise RuntimeError("Preflight failed")

# Load model (using transformers)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create optimizer
optimizer = ChimeraGearOptimizer(
    model.parameters(),
    lr=3e-4,
    gear='auto',
    cpu_offload=True,
    bo_enabled=True
)

# Training
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        
        # Optimizer handles gear shifts automatically
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"Gear: {optimizer.current_gear}")
```

### Manual Gear Management

```python
from chimera_m import ChimeraGearOptimizer

optimizer = ChimeraGearOptimizer(
    model.parameters(),
    gear=2,  # Start at Level 2
    auto_gear=False  # Disable watchdog
)

# Manual gear changes
for epoch in range(epochs):
    if epoch == 5:
        # Compress further
        optimizer.apply_gear_compression(3)
        print(f"Manually shifted to gear {optimizer.current_gear}")
    
    # ... training ...
```

### Custom Watchdog

```python
from chimera_m import GearshiftWatchdog, GEAR_LEVELS

# Custom thresholds
watchdog = GearshiftWatchdog(
    poll_interval_ms=50.0,  # Faster polling
    ram_threshold=0.90,     # Higher RAM threshold
    bo_enabled=True
)

# Custom BO
from chimera_m import BayesianOptimizer

watchdog.bo = BayesianOptimizer(
    kernel_type='rbf',  # Smoother kernel
    noise_variance=0.02  # Less uncertainty
)
watchdog.bo.add_param('vram_threshold', 0.75, 0.95)
watchdog.bo.add_param('hold_duration', 45.0, 180.0)

# Use with optimizer
optimizer = ChimeraGearOptimizer(
    model.parameters(),
    gear='auto',
    watchdog=watchdog
)
```

## Dataset Formatting

### Plain Text → Chat Format

Input (`Datasets/data.txt`):
```
This is a sample text about machine learning.

Another paragraph here with different content.
```

Auto-format output:
```
[WARNING] Dataset format mismatch detected
  Model format: llama (requires chat)
  Input format: text
[APPLYING FIX] Wrapping with system prompt
[RESULT] Created 2 conversation samples
```

Result:
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "This is a sample text about machine learning."}
]}
```

### JSONL Validation

Input (`Datasets/data.jsonl`):
```json
{"text": "Sample text"}
{"text": "Another sample"}
```

For Llama model, system will:
1. Detect mismatch (text vs chat)
2. Wrap each text as user message
3. Add system prompt

## Checkpoint Management

### Gear Transition Checkpoints

```python
from chimera_m import load_checkpoint

# After automatic gear shift
# Files created:
# - gear_transition_step500_pre.pt
# - gear_transition_step500_post.pt

# Load pre-transition state
checkpoint = torch.load("gear_transition_step500_pre.pt")
print(f"Gear before: {checkpoint['gear']}")  # 2

# Load post-transition state  
checkpoint = torch.load("gear_transition_step500_post.pt")
print(f"Gear after: {checkpoint['gear']}")  # 3
```

### Custom Checkpoint Intervals

```bash
# Save every 1000 steps
python train.py --checkpoint-interval 1000
```

## Debugging

### Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from chimera_m import main
main()
```

### Monitor Gear Shifts

```python
from chimera_m import ChimeraGearOptimizer

optimizer = ChimeraGearOptimizer(model.parameters())

# Track shifts
shift_history = []
original_execute = optimizer.watchdog.execute_shift

def track_shift(old_gear, new_gear):
    shift_history.append({
        'step': optimizer.step_count,
        'from': old_gear,
        'to': new_gear,
        'timestamp': time.time()
    })
    original_execute(new_gear)

optimizer.watchdog.execute_shift = track_shift
```

### Memory Profiling

```python
import torch
from chimera_m import ChimeraGearOptimizer

optimizer = ChimeraGearOptimizer(model.parameters(), gear=4)

# Profile each gear
for gear in [1, 2, 3, 4, 5]:
    optimizer.apply_gear_compression(gear)
    
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Gear {gear}: {peak:.2f}GB peak VRAM")
```

## Integration Examples

### With HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from chimera_m import ChimeraGearOptimizer

# Create optimizer
optimizer = ChimeraGearOptimizer(model.parameters(), gear='auto')

# Use in Trainer
training_args = TrainingArguments(
    output_dir="./Output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    optimizers=(optimizer, None)  # (optimizer, lr_scheduler)
)

trainer.train()
```

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from chimera_m import ChimeraGearOptimizer

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ...
    
    def configure_optimizers(self):
        return ChimeraGearOptimizer(
            self.parameters(),
            lr=1e-4,
            gear='auto'
        )
```

## Performance Tuning

### Optimize for Speed

```bash
# Lower compression, faster training
python train.py --gear 2 --batch-size 4
```

### Optimize for Memory

```bash
# Higher compression, lower memory
python train.py --gear 5 --max-length 256
```

### Balance

```bash
# Middle ground
python train.py --gear 3 --batch-size 2 --max-length 512
```
