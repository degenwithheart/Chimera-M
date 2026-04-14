# CHIMERA-M: Compressive Hybrid Architecture for Intelligent, Efficient Resource Allocation and Modeling

**CHIMERA-M** is a self-contained, drop-in training system featuring autonomous CPU↔GPU adaptation, extreme compression (5× to 50×), and Bayesian-optimized gearshift watchdog. It automatically detects hardware, formats datasets, and adjusts compression levels in real-time without stopping training.

## Key Features

- **5 Compression Levels**: From BF16 (5×) to MEZO+SSD (50×)
- **Autonomous Gearshift**: GPU-only watchdog with 100ms polling, adaptive Bayesian optimization
- **Auto-Formatting**: Detects and formats datasets to match model requirements
- **Never Stops Training**: Emergency downshift to Level 5 on OOM
- **Self-Contained**: Single Python file, no external imports required

## Quick Start

```bash
mkdir -p Models Datasets Output
python train.py
```

See **[Usage Guide](docs/USAGE.md)** for detailed setup, installation, and workflows.

## Documentation

| Document | Contents |
|----------|----------|
| **[Architecture](docs/ARCHITECTURE.md)** | System design and data flow |
| **[API Reference](docs/API_REFERENCE.md)** | All classes and functions |
| **[Configuration](docs/CONFIGURATION.md)** | Gear levels, thresholds, tuning |
| **[Examples](docs/EXAMPLES.md)** | Code samples and scenarios |
| **[Usage Guide](docs/USAGE.md)** | Installation, setup, running training, workflows |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and fixes |
| **[C/C++ Extensions](chimera_m_cpp/README.md)** | Optional acceleration extensions (10-100× speedup) |
| **[Adapter Architecture](docs/ADAPTER_ARCHITECTURE.md)** | *Roadmap v1.2: LoRA/QLoRA/Unsloth integration design* |
| **[TriAttention Architecture](docs/TRIATTENTION_ARCHITECTURE.md)** | *Roadmap v1.3: KV cache compression for inference* |
| **[Multi-GPU Architecture](docs/MULTI_GPU_ARCHITECTURE.md)** | *Roadmap v1.4+: Future distributed training design* |

## Project Structure

```
chimera-m/
├── chimera_m.py          # Core system (~2300 lines, self-contained)
├── train.py              # Entry point (4 lines)
├── chimera_m_cpp/        # C++ extensions (optional, pybind11)
├── chimera_m_c/          # C extensions (optional, ctypes)
├── Models/               # Place 1 model here
├── Datasets/             # Place dataset(s) here
├── Output/               # Auto-created: checkpoints, logs
└── docs/                 # Documentation (9 files)
```

## CLI Quick Reference

```bash
python train.py --epochs 50 --lr 2e-4 --gear auto
python train.py --gear 3 --batch-size 2
python train.py --resume ./Output/checkpoint_epoch5.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 1 | Batch size |
| `--max-length` | 512 | Max sequence length |
| `--gear` | auto | Compression: 1-5 or auto |
| `--resume` | - | Checkpoint to resume |

## Gear Levels

| Level | Compression | Use Case |
|-------|-------------|----------|
| 1 | 5× | Model fits in VRAM |
| 2 | 10× | Balanced |
| 3 | 20× | 8B on 24GB |
| 4 | 40× | RAM overflow → SSD |
| 5 | 50× | Emergency/CPU-only |

## Roadmap

- **Current**: Single-GPU training with 5 adaptive compression levels
- **Next**: Optional C/C++ extensions for 10-100× speedup on serialization
- **Future v1.2**: LoRA/QLoRA adapter support with per-layer gear assignment
  - *Status: Architecture drafted, pending implementation*
  - See [Adapter Architecture](docs/ADAPTER_ARCHITECTURE.md) for design
- **Future v1.3**: TriAttention KV cache compression for unified training+inference
  - *Status: Architecture drafted, pending implementation*
  - See [TriAttention Architecture](docs/TRIATTENTION_ARCHITECTURE.md) for design
- **Future v1.4+**: Multi-GPU distributed training with asymmetric gear assignment
  - *Status: Architecture drafted, pending efficient communication solutions*
  - See [Multi-GPU Architecture](docs/MULTI_GPU_ARCHITECTURE.md) for design

## License

MIT License
