# API Reference

Complete API documentation for all classes and functions.

## Table of Contents
- [BayesianOptimizer](#bayesianoptimizer)
- [TernaryCodec](#ternarycodec)
- [CountMinSketch](#countminsketch)
- [PagedMemory](#pagedmemory)
- [GearshiftWatchdog](#gearshiftwatchdog)
- [ChimeraGearOptimizer](#chimeragearoptimizer)
- [Validation Functions](#validation-functions)

---

## BayesianOptimizer

Bayesian optimization using Gaussian Process.

```python
class BayesianOptimizer(
    kernel_type: str = 'matern',
    length_scale: float = 1.0,
    variance: float = 1.0,
    noise_variance: float = 0.05,
    acquisition_type: str = 'ei',
    xi: float = 0.01
)
```

### Methods

#### `add_param(name, low, high)`
Add parameter with bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Parameter identifier |
| `low` | float | Lower bound |
| `high` | float | Upper bound |

**Example:**
```python
bo = BayesianOptimizer()
bo.add_param('vram_threshold', 0.70, 0.95)
bo.add_param('hold_duration', 30.0, 120.0)
```

#### `suggest(rng=None) -> Dict[str, float]`
Suggest next parameters to evaluate.

**Returns:** Dictionary mapping parameter names to values.

**Behavior:**
- First 10 calls: Random exploration
- After 10: Acquisition function optimization

#### `update(params, value)`
Update with observed objective.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | Dict[str, float] | Parameters that were evaluated |
| `value` | float | Objective value (lower = better) |

#### `get_best() -> Tuple[Dict, float]`
Get best parameters found.

**Returns:** `(best_params, best_objective)`

---

## TernaryCodec

Quantizes weights to {-1, 0, +1}.

```python
class TernaryCodec(stochastic: bool = True)
```

### Methods

#### `encode(tensor, seed=None) -> Tuple[Tensor, Tuple]`
Encode to packed ternary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | torch.Tensor | Input tensor (any shape) |
| `seed` | int | Optional seed for stochastic rounding |

**Returns:** `(packed_tensor, metadata)`

**Compression:** ~16× (2 bits vs 32 bits)

**Example:**
```python
codec = TernaryCodec(stochastic=True)
tensor = torch.randn(1000, 1000)
packed, meta = codec.encode(tensor, seed=42)
# packed.shape: (62500,) int32
```

#### `decode(packed, metadata) -> Tensor`
Decode back to floating-point.

| Parameter | Type | Description |
|-----------|------|-------------|
| `packed` | torch.Tensor | From encode() |
| `metadata` | Tuple | From encode() |

**Returns:** Reconstructed tensor

---

## CountMinSketch

Constant-memory optimizer states.

```python
class CountMinSketch(width: int = 1024, depth: int = 4, device='cuda')
```

**Memory:** 32KB constant (2 × depth × width × 4 bytes)

### Methods

#### `update(indices, values_m, values_v, beta1=0.9, beta2=0.999)`
Update sketch with gradient statistics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `indices` | torch.Tensor | Flat parameter indices |
| `values_m` | torch.Tensor | Momentum values |
| `values_v` | torch.Tensor | Variance values |
| `beta1` | float | Momentum decay |
| `beta2` | float | Variance decay |

#### `query(indices) -> Tuple[Tensor, Tensor]`
Query optimizer state.

**Returns:** `(momentum_estimate, variance_estimate)`

**Example:**
```python
sketch = CountMinSketch(width=1024, depth=4)
grad = torch.randn(10000)
indices = torch.arange(10000)

# Update
sketch.update(indices, grad.abs(), grad ** 2)

# Query
m_hat, v_hat = sketch.query(indices[:100])
update = m_hat / (v_hat.sqrt() + 1e-8)
```

---

## PagedMemory

SSD offloading for RAM overflow.

```python
class PagedMemory(
    page_size_mb: int = 256,
    ram_threshold: float = 0.85,
    cache_dir: str = "./Output/ssd_cache"
)
```

### Methods

#### `check_ram_pressure() -> bool`
Check if RAM exceeds threshold.

**Returns:** True if `ram_used / ram_total > ram_threshold`

#### `spill_to_ssd(param_id, tensor)`
Move tensor to SSD.

| Parameter | Type | Description |
|-----------|------|-------------|
| `param_id` | str | Unique identifier |
| `tensor` | torch.Tensor | Tensor to spill |

**Process:** numpy → pickle → lz4 → disk

#### `load_from_ssd(param_id) -> Tensor`
Load tensor from SSD.

**Returns:** Tensor on original device

#### `get_tensor(param_id) -> Optional[Tensor]`
Get tensor (from RAM or SSD).

#### `cleanup()`
Remove all page files.

---

## GearshiftWatchdog

Real-time gear monitoring.

```python
class GearshiftWatchdog(
    poll_interval_ms: float = 100.0,
    cpu_pin: int = 0,
    ram_threshold: float = 0.85,
    bo_enabled: bool = True
)
```

**Memory:** <1KB (L2 resident ring buffers)

### Methods

#### `start()`
Start monitoring thread.

#### `stop()`
Stop monitoring thread.

#### `should_shift(current_loss=None) -> Tuple[bool, int]`
Check if gear change needed.

**Returns:** `(should_shift, target_gear)`

**Logic:**
- Checks hold period first (must expire)
- VRAM pressure → downshift
- Stable VRAM + improving loss → upshift

#### `execute_shift(new_gear, callback=None)`
Perform gear change.

| Parameter | Type | Description |
|-----------|------|-------------|
| `new_gear` | int | Target gear 1-5 |
| `callback` | Callable | `callback(old_gear, new_gear)` |

#### `emergency_downshift(callback=None)`
Force immediate downshift to Level 5.

#### `update_objective(step_time, oom_risk, loss_plateau)`
Record for BO optimization.

---

## ChimeraGearOptimizer

Main optimizer class.

```python
class ChimeraGearOptimizer(
    params,
    lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    gear: Union[int, str] = 'auto',
    device: str = 'cuda',
    cpu_offload: bool = True,
    ssd_offload: bool = False,
    ram_threshold: float = 0.85,
    mezo_mode: bool = False,
    bo_enabled: bool = True
)
```

### Methods

#### `step(closure=None) -> Optional[float]`
Perform optimization step.

**Parameters:**
- `closure`: Optional callable returning loss

**Process:**
1. Check gear shift
2. Execute level-appropriate update
3. Return loss

#### `state_dict() -> Dict`
Save state.

**Includes:**
- Optimizer state
- Current gear
- Shadow weights
- Error feedback

#### `load_state_dict(state_dict)`
Restore state.

#### `_apply_gear_compression(new_gear)` (internal)
Change compression level. This is an internal method used by the gearshift watchdog.

**Creates checkpoints:**
- `gear_transition_step{N}_pre.pt`
- `gear_transition_step{N}_post.pt`

#### `emergency_downshift()`
Force Level 5 on OOM.

---

## Validation Functions

### `preflight_check(model_path, dataset_files, gear, args) -> Dict`

Validate before training.

**Checks:**
- Hardware (CUDA, VRAM ≥ 4GB, RAM)
- Model (exists, not too large)
- Dataset (exists, UTF-8, readable)
- Dependencies (lz4, transformers)
- Permissions (output dir writable)
- Config (LR, batch size valid)

**Returns:**
```python
{
    'passed': bool,
    'errors': List[str],
    'warnings': List[str],
    'hardware': Dict,
    'model': Dict,
    'dataset_stats': Dict
}
```

### `print_preflight_report(results)`
Print formatted report.

---

## Gear Levels Reference

| Level | Weights | Optimizer | Shadow | SSD | Ratio |
|-------|---------|-----------|--------|-----|-------|
| 1 | BF16 | FP8 | No | No | 5× |
| 2 | Ternary | Sketch | Yes | No | 10× |
| 3 | Ternary | Sparse Sketch | Yes | No | 20× |
| 4 | Ternary | Sparse Sketch | Yes | Yes | 40× |
| 5 | MEZO | Minimal | Yes | Yes | 50× |
