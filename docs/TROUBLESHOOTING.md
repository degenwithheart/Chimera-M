# Troubleshooting

Common issues and solutions.

## Installation Issues

### "No module named 'lz4'"

**Error:**
```
ModuleNotFoundError: No module named 'lz4'
```

**Fix:**
```bash
pip install lz4
```

**When Required:** Gear levels 4-5 (SSD offloading)

### "No module named 'transformers'"

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Fix:**
```bash
pip install transformers
```

**When Required:** Model auto-detection

### CUDA Not Available

**Error:**
```
CUDA not available
```

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix:**
```bash
# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Preflight Failures

### "Model too large"

**Error:**
```
Model too large: 8.0B params requires 25.6GB VRAM even at 50× compression
Detected: 24.0GB
Use smaller model or CPU-only training
```

**Solution:**
```bash
# Option 1: Use CPU (very slow)
python train.py --gear 5

# Option 2: Use smaller model
# Download 3B or 7B variant instead

# Option 3: Reduce max sequence length
python train.py --max-length 256
```

### "No model found"

**Error:**
```
No model found in ./Models
Please place your model in the Models/ folder
```

**Fix:**
```bash
mkdir -p Models
cp -r /path/to/model Models/
```

### "No datasets found"

**Error:**
```
No datasets found in ./Datasets
Supported: .jsonl, .json, .txt, .csv, .parquet
```

**Fix:**
```bash
mkdir -p Datasets
cp /path/to/data.jsonl Datasets/
```

### "Permission denied"

**Error:**
```
Permission denied: ./Datasets/data.jsonl
```

**Fix:**
```bash
chmod 644 Datasets/*
chmod 755 Output
```

## Runtime Errors

### OOM During Training

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Automatic Fix:**
Chimera-M should auto-downshift to Level 5. If not:

**Manual Fix:**
```bash
# Start with higher compression
python train.py --gear 5 --batch-size 1 --max-length 256
```

**Reduce Memory:**
- Lower `--max-length` (512 → 256)
- Reduce `--batch-size` (must be ≥1)
- Increase compression level (--gear 5)

### SSD Spill Fails

**Symptom:**
```
[WARNING] SSD spill failed, RAM may overflow
```

**Check Disk Space:**
```bash
df -h .
```

**Fix:**
```bash
# Clear old checkpoints
rm Output/*.pt

# Or use different output directory
python train.py --output-dir /mnt/large_disk/Output
```

### Watchdog Not Starting

**Symptom:** No gear shifts during training

**Causes:**
1. CUDA not available
2. `--bo-off` flag used
3. Manual gear selected

**Check:**
```python
# In Python
from chimera_m import ChimeraGearOptimizer

optimizer = ChimeraGearOptimizer(model.parameters(), gear='auto')
print(optimizer.watchdog)  # Should not be None
```

### Stuck at Gear Level

**Symptom:** Never upshifts or downshifts

**Causes:**
1. In hold period (60 seconds)
2. BO hasn't learned yet (first 100 steps)
3. Thresholds misconfigured

**Force Shift:**
```python
# In code
optimizer.apply_gear_compression(3)  # Force gear 3
```

### Training Extremely Slow

**Symptom:** < 1 step per second

**Causes:**
1. CPU-only with large model
2. SSD thrashing (Level 4)
3. MEZO mode (Level 5, 2× forward passes)

**Diagnose:**
```bash
# Check current gear
# Look for "Gear: X" in output
```

**Fix:**
```bash
# If gear 4/5 on CPU, use GPU or smaller model
# If SSD thrashing, increase RAM or reduce model size

# Monitor SSD
iostat -x 1  # Check %util (should be < 80%)
```

## Data Issues

### Dataset Format Mismatch

**Symptom:**
```
[WARNING] Dataset format mismatch detected
```

**Expected:** Auto-format should handle this

**If Fails:**
```python
# Manual format
from chimera_m import auto_format_dataset, infer_dataset_format
from pathlib import Path

files = [Path("Datasets/data.txt")]
source = infer_dataset_format(files)
target = "chat"  # or "text"

formatted = auto_format_dataset(files, source, target)
# Save formatted data
```

### UTF-8 Encoding Errors

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Fix:**
```bash
# Convert encoding
iconv -f ISO-8859-1 -t UTF-8 Datasets/data.txt > Datasets/data_utf8.txt
```

### Invalid JSON Lines

**Error:**
```
JSONDecodeError: Expecting property name
```

**Fix:**
```bash
# Validate JSONL
python -c "
import json
with open('Datasets/data.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Line {i+1} invalid: {line[:50]}')
"
```

## Checkpoint Issues

### Resume Fails

**Error:**
```
Cannot read checkpoint: File not found
```

**Fix:**
```bash
# Verify checkpoint exists
ls -la Output/checkpoint_epoch*.pt

# Resume with full path
python train.py --resume Output/checkpoint_epoch5.pt
```

### Gear Mismatch on Resume

**Warning:**
```
Resume checkpoint was at gear 3, but you specified gear 2
Will use checkpoint's gear level
```

**This is normal** - system preserves saved gear to maintain state consistency.

### Checkpoint Corruption

**Error:**
```
RuntimeError: Invalid magic number; corrupt file?
```

**Fix:**
```bash
# Use earlier checkpoint
python train.py --resume Output/checkpoint_epoch4.pt

# Or start fresh
rm Output/*.pt
python train.py
```

## Performance Issues

### VRAM Not Fully Used

**Symptom:** Using 12GB of 24GB, but slow

**Solution:**
```bash
# Increase batch size
python train.py --batch-size 2  # or 4

# Or increase sequence length
python train.py --max-length 1024
```

### GPU Underutilized

**Symptom:** GPU utilization < 50% in nvidia-smi

**Causes:**
1. CPU bottleneck (data loading)
2. Small batch size
3. SSD thrashing (Level 4)

**Fix:**
```bash
# Pre-load dataset to RAM
# Increase batch size
# Reduce compression level (--gear 2 or 3)
```

### High Step Time Variance

**Symptom:** Step time varies 0.5s to 5s

**Causes:**
1. Gear shifts happening
2. SSD paging
3. System load

**Check:**
```bash
# Watch for shifts in output
grep "Gear shift" Output/training.log
```

## BO (Bayesian Optimization) Issues

### BO Not Improving

**Symptom:** Same thresholds after 200+ steps

**Check:**
```python
# In code
print(optimizer.watchdog.bo.get_best())
```

**Possible Causes:**
1. Not enough observations (need 10+)
2. All objectives similar (no learning signal)
3. Stuck in local minimum

**Fix:**
```bash
# Disable BO, use fixed thresholds
python train.py --bo-off
```

### BO Suggests Invalid Values

**Symptom:** `vram_threshold > 1.0` or negative

**Fix:** Already handled by clipping in `Bounds.clip()`

## Hardware-Specific

### L4 24GB Issues

**Issue:** 8B model OOMs randomly

**Expected:** Level 3 should handle it, but marginal

**Solution:**
```bash
# Force Level 4 from start
python train.py --gear 4

# Or reduce max length
python train.py --max-length 384
```

### CPU Training Too Slow

**Expected:** 250M model = ~1 token/sec

**Solutions:**
```bash
# Use SSD offload earlier
python train.py --gear 4 --ssdr-threshold 0.70

# Smaller model
# Cloud GPU instance
```

### Multiple GPUs

**Issue:** Only using GPU 0

**Current:** Chimera-M is single-GPU only

**Workaround:**
```bash
# Select different GPU
CUDA_VISIBLE_DEVICES=1 python train.py
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from chimera_m import main
main()
```

### Monitor Memory

```python
import torch
from chimera_m import ChimeraGearOptimizer

optimizer = ChimeraGearOptimizer(model.parameters())

for step in range(100):
    # Training...
    
    if step % 10 == 0:
        print(f"Step {step}:")
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"  Gear: {optimizer.current_gear}")
```

### Profile Step Time

```python
import time

optimizer = ChimeraGearOptimizer(model.parameters())

times = []
for _ in range(100):
    start = time.time()
    optimizer.step()
    times.append(time.time() - start)

print(f"Mean: {sum(times)/len(times):.3f}s")
print(f"Std:  {np.std(times):.3f}s")
print(f"Max:  {max(times):.3f}s")
```

### Check Data Loading

```python
from chimera_m import scan_datasets, infer_dataset_format

files = scan_datasets("./Datasets")
print(f"Found {len(files)} files")

for f in files:
    fmt = infer_dataset_format([f])
    print(f"{f.name}: {fmt}")
```

## Getting Help

### Gather Info

Before reporting issue:

```bash
# Hardware
python -c "
import torch
from chimera_m import detect_hardware
hw = detect_hardware()
for k, v in hw.items():
    print(f'{k}: {v}')
"

# Versions
python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
"

# Preflight output
python train.py 2>&1 | head -100
```

### Common Command Template

```bash
# Debug run with maximum info
python train.py \
    --epochs 1 \
    --batch-size 1 \
    --max-length 128 \
    --gear auto \
    --log-interval 1 \
    2>&1 | tee debug.log
```

## Known Limitations

1. **Single GPU Only** - No multi-GPU support
2. **FP16/BF16 Only** - No FP32 training
3. **PyTorch Only** - No JAX/TensorFlow
4. **Transformers Models** - Custom architectures need manual loading
5. **Watchdog GPU-Only** - No dynamic adaptation on CPU-only

## FAQ

**Q: Why does gear shift cause pause?**
A: Recompression and checkpointing takes 1-5 seconds. Normal.

**Q: Can I disable checkpointing on shift?**
A: No, safety feature. But you can delete old checkpoints manually.

**Q: Why 60 second hold?**
A: Prevents thrashing. Stabilizes before next decision.

**Q: Can I use my own optimizer?**
A: No, ChimeraGearOptimizer replaces standard optimizer completely.

**Q: Does it work with DeepSpeed/FSDP?**
A: No, incompatible architectures. Use Chimera-M standalone.

**Q: Can I train from scratch?**
A: Yes, but pre-trained models recommended for convergence.
