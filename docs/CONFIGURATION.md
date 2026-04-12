# Configuration Guide

All configuration options and gear level details.

## Command-Line Arguments

### Training Parameters

| Argument | Default | Range | Description |
|----------|---------|-------|-------------|
| `--epochs` | 10 | ≥1 | Training epochs |
| `--lr` | 3e-4 | 1e-6 to 1e-1 | Learning rate |
| `--batch-size` | 1 | ≥1 | Samples per batch |
| `--max-length` | 512 | ≥1 | Max sequence length |

### Gear Parameters

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--gear` | auto | 1-5, auto | Compression level |
| `--ssdr-threshold` | 0.85 | 0.0-1.0 | RAM % to trigger SSD |
| `--bo-off` | False | flag | Disable BO |

### Path Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--models-dir` | ./Models | Model location |
| `--datasets-dir` | ./Datasets | Dataset location |
| `--output-dir` | ./Output | Checkpoints/logs |
| `--resume` | None | Checkpoint to resume |

## Gear Level Details

### Level 1: Standard (5×)

**Use When:** Model fits comfortably in VRAM

**Memory Layout:**
- Weights: BF16 on GPU (2 bytes/param)
- Optimizer: FP8 on GPU (1 byte/param)
- Activations: BF16

**Update:** Standard AdamW

**Speed:** 100% (baseline)

### Level 2: Balanced (10×)

**Use When:** Model fits but optimizer states cause pressure

**Memory Layout:**
- Weights: Ternary on GPU (0.33 bytes/param packed)
- Optimizer: Count-Min Sketch constant 16KB
- Shadow: BF16 on CPU (2 bytes/param)

**Update:**
1. Compute gradient on GPU ternary weights
2. Update sketch with gradient stats
3. Query sketch for momentum/variance
4. Update CPU shadow weights
5. Requantize shadow → GPU ternary

**Speed:** 80-90%

### Level 3: Compressed (20×)

**Use When:** 8B model on 24GB GPU (edge case)

**Memory Layout:**
- Same as Level 2

**Update:**
- Sparse gradients: Keep only top 0.1% by magnitude
- Remaining steps same as Level 2

**Speed:** 70-80%

### Level 4: SSD Offload (40×)

**Use When:** Shadow weights exceed RAM

**Memory Layout:**
- Weights: Ternary on GPU
- Optimizer: Sketch on GPU
- Shadow: BF16 on CPU (active) + SSD (spilled)

**Spilling:**
- Trigger: `ram_used / ram_total > 0.85`
- Largest shadows spilled first
- 256-512MB pages (adaptive)
- LZ4 compression (2-5×)

**Speed:** 30-50% (SSD bottleneck)

### Level 5: MEZO Extreme (50×)

**Use When:** Emergency OOM or CPU-only training

**Memory Layout:**
- Weights: Ternary on GPU
- Optimizer: None (minimal state)
- No momentum tracking

**Update (Zeroth-Order):**
1. Save current weights
2. Perturb: `w' = w + ε · z` where z ~ Rademacher
3. Forward pass with w' → loss⁺
4. Forward pass with w → loss⁻
5. Gradient estimate: `(loss⁺ - loss⁻) / (2ε) · z`
6. Update: `w = w - lr · estimate`

**Speed:** 10-30% (2 forward passes per step)

**Benefits:**
- No backward pass memory
- No optimizer state
- Works with 1 sample

## Watchdog Thresholds

### Bayesian Optimization Parameters

| Parameter | Default Range | Description |
|-----------|---------------|-------------|
| `vram_downshift` | 0.70-0.95 | VRAM % to trigger downshift |
| `loss_spike` | 1.1-3.0 | Loss increase factor for pressure |
| `hold_duration` | 30-120s | Minimum time at gear level |

**Objective:** `minimize(total_time + 10×oom_events + 5×loss_plateaus)`

### Fixed Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poll_interval` | 100ms | Watchdog check frequency |
| `upsift_hysteresis` | 0.15 | VRAM % below threshold to upshift |
| `pressure_weight_vram` | 0.4 | VRAM contribution to pressure score |
| `pressure_weight_loss` | 0.3 | Loss trend contribution |
| `pressure_weight_variance` | 0.2 | Step time variance contribution |
| `pressure_threshold` | 0.5 | Score to trigger shift |

## Hardware-Based Auto-Selection

Algorithm for `--gear auto`:

```
model_bytes_bf16 = num_params × 2
vram_bytes = vram_gb × 1024³

IF model_bytes_bf16 × 1.5 < vram_bytes:
    RETURN 1

ELIF model_bytes_bf16 / 6 < vram_bytes:
    RETURN 2

ELIF model_bytes_bf16 / 20 < vram_bytes:
    RETURN 3

ELIF model_bytes_bf16 / 40 < vram_bytes AND ram_gb > 16:
    RETURN 4

ELSE:
    RETURN 5
```

## Compression Ratios

| Component | Uncompressed | Compressed | Ratio |
|-----------|--------------|------------|-------|
| Weights (FP32) | 4 bytes | 0.33 bytes (ternary) | 12× |
| Weights (BF16) | 2 bytes | 0.33 bytes | 6× |
| Optimizer (Adam) | 8 bytes | 16KB constant | ∞ (for large models) |
| Gradients | 4 bytes | 0.004 bytes (0.1% sparse) | 1000× |

**Effective Ratios:**
- Level 1: 5× (BF16 + FP8)
- Level 2: 10× (ternary + sketch)
- Level 3: 20× (ternary + sparse sketch)
- Level 4: 40× (level 3 + SSD offload multiplier)
- Level 5: 50× (MEZO minimal state)

## SSD Page Sizing

Automatic based on DDR:

| DDR Version | Speed | Page Size |
|-------------|-------|-----------|
| DDR4 | 2133-2666 | 256MB |
| DDR4 | 3200-3600 | 384MB |
| DDR5 | 4800+ | 512MB |

Larger pages = fewer IOPS but higher throughput
Smaller pages = more IOPS, better for random access
