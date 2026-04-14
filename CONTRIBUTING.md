# Contributing to CHIMERA-M

**Compressive Hybrid Architecture for Intelligent, Efficient Resource Allocation and Modeling**

Single-file training system with autonomous compression gearshift. All core algorithms implemented.

---

## Current Architecture

```
chimera-m/
├── chimera_m.py      # Core system (~3000 lines, self-contained)
├── train.py          # 4-line entry point
├── requirements.txt  # 5 dependencies
├── docs/             # 6 documentation files
├── Models/           # User model directory (empty initially)
├── Datasets/         # User dataset directory (empty initially)
└── Output/           # Auto-created for checkpoints
```

All functionality lives in `chimera_m.py`. No subdirectories, no separate modules.

---

## Priority Contributions

### 🔴 CRITICAL: GPU Benchmarking

**Status:** Code complete. Performance claims need validation.

**What to measure:**
- Speed vs AdamW baseline on 3B/7B models
- Memory usage at each gear level (1-5)
- Convergence curves (loss over 1000+ steps)
- Gear shift frequency and stability

**How to contribute:**
1. Run training on your hardware
2. File GitHub Issue with `[BENCHMARK]` label
3. Include:
   - GPU model, CUDA version, driver version
   - Model size and name
   - Gear level used
   - Wall time per step
   - Peak memory usage
   - Final loss value

---

### 🟡 High Priority

#### Test Coverage
Currently minimal tests. Need:
- Unit tests for each component class
- Integration tests for full training loop
- Edge case tests (OOM handling, gear transitions)
- Hardware detection validation

**File:** Create `tests/test_chimera_m.py`

#### Examples
- Training scripts for specific models (Llama-3B, Phi-2, Qwen)
- Custom dataset format examples
- Resume from checkpoint examples

**File:** Create `examples/train_llama.py`, etc.

#### Bug Fixes
- Report and fix any crashes or OOMs
- Numerical stability issues
- Dataset format detection edge cases

---

### 🟢 Medium Priority

#### Documentation Improvements
- Clarify docstrings in complex functions
- Add inline comments for math-heavy sections
- Improve error messages

#### Hardware Support
- AMD ROCm compatibility testing
- Apple Metal (MPS) backend
- Intel Arc/Xe testing

---

## Code Standards

✅ **Self-contained changes** (single file only)  
✅ **Docstrings** for new classes/functions  
✅ **Type hints** where feasible  
✅ **No external dependencies** (use existing: torch, numpy, psutil, lz4)  
✅ **Error handling** for edge cases  

## Contribution Process

1. Fork repository
2. Create branch: `git checkout -b feature/name`
3. Make changes to `chimera_m.py` or `train.py`
4. Test locally
5. Submit PR with:
   - What changed
   - Why it matters
   - How you tested
   - Hardware tested on

---

## Entry Level (Good First PRs)

1. **Documentation fixes** (~30 min)
   - Fix typos, clarify docstrings
   - Add comments to unclear code sections

2. **Examples** (~1-2 hours)
   - Create `examples/train_phi2.py`
   - Show specific use case

3. **Tests** (~2-4 hours)
   - Test ternary quantization
   - Test Count-Min Sketch accuracy
   - Test gear transitions

**Getting started:**
```bash
git clone https://github.com/degenwithheart/chimera-m.git
cd chimera-m
pip install -r requirements.txt
python train.py --epochs 1 --max-length 128  # Quick test
```

## Intermediate Level

1. **Optimization** (~1-2 weeks)
   - Profile hot paths in `chimera_m.py`
   - Optimize ternary packing/unpacking
   - Improve SSD I/O patterns

2. **Hardware backends** (~2-4 weeks)
   - ROCm compatibility
   - Metal Performance Shaders
   - Intel oneAPI

## Advanced Level

1. **Multi-GPU support** (~4-6 weeks)
   - FSDP integration
   - Distributed gearshift coordination
   - Gradient aggregation

2. **Inference optimization** (~2-3 weeks)
   - Fast ternary matmul kernels
   - KV-cache optimization

---

## Project Structure Details

### `chimera_m.py` organization

```python
# Section 1: Bayesian Optimizer (lines 67-500)
# - Kernel classes (RBF, Matérn)
# - GaussianProcess
# - AcquisitionFunction
# - BayesianOptimizer

# Section 2: Ternary Codec (lines 501-700)
# - TernaryCodec (pack/unpack)
# - Stochastic rounding

# Section 3: Count-Min Sketch (lines 701-900)
# - CountMinSketch (16KB optimizer state)
# - Hash functions

# Section 4: Paged Memory (lines 901-1100)
# - PagedMemory (SSD offloading)
# - LZ4 compression
# - Page eviction

# Section 5: Gearshift Watchdog (lines 1101-1400)
# - GearshiftWatchdog (100ms polling)
# - Pressure detection
# - BO threshold tuning

# Section 6: Main Optimizer (lines 1401-2000)
# - ChimeraGearOptimizer (5 levels)
# - Step functions per level
# - State management

# Section 7: Training Utilities (lines 2001-2349)
# - Hardware detection
# - Dataset formatting
# - Preflight checks
# - Main training loop
```

---

## Code of Conduct

- Be respectful
- Be helpful
- Be honest about capabilities
- Be patient with review process

---

## Resources

- [README.md](README.md) - Overview
- [docs/USAGE.md](docs/USAGE.md) - Setup guide
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - API docs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Design

---

## Contact

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** Questions, general chat

---

**Thank you for contributing to CHIMERA-M!**
