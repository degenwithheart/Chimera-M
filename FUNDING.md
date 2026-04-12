# CHIMERA-M: Support & Funding

**Compressive Hybrid Architecture for Intelligent, Efficient Resource Allocation and Modeling**

A single-file, self-contained training system with autonomous compression gearshift.

---

## Current Status

- **Version:** 1.0.0
- **Development:** Solo developer
- **Code:** Single-file architecture (`chimera_m.py`, ~2300 lines)
- **License:** MIT (free to use)

---

## What's Implemented

✅ **Core Optimizer** - 5-level compression (5× to 50×)  
✅ **Ternary Quantization** - Weights to {-1, 0, +1}  
✅ **Count-Min Sketch** - 16KB optimizer state  
✅ **Bayesian Optimizer** - Gearshift threshold tuning  
✅ **Gearshift Watchdog** - GPU monitoring, auto up/downshift  
✅ **Paged Memory** - SSD offloading for gears 4-5  
✅ **MEZO** - Zeroth-order fallback (level 5)  
✅ **Auto-formatting** - Dataset detection and conversion  
✅ **Preflight checks** - Hardware/model validation  

---

## What Would Benefit from Support

| Task | Effort | Benefit |
|------|--------|----------|
| GPU benchmarking suite | 2-3 weeks | Validated speedup claims |
| AMD ROCm support | 4-6 weeks | AMD GPU compatibility |
| Apple Metal support | 4-6 weeks | M1/M2/M3 Macs |
| Intel Arc/Xe support | 4-6 weeks | Intel GPU compatibility |
| Multi-GPU (FSDP) | 6-8 weeks | Distributed training |
| Inference optimization | 4-6 weeks | Fast ternary matmul |

---

## Sponsorship

CHIMERA-M is MIT licensed and free. Support development via:

**GitHub Sponsors:** [github.com/sponsors/degenwithheart](https://github.com/sponsors/degenwithheart)

- **$5/month** - README acknowledgment
- **$25/month** - Priority issue response  
- **$100+/month** - Feature prioritization

---

## Citation

```bibtex
@software{chimera_m_2026,
  title={CHIMERA-M: Compressive Hardware-Intelligent Gearshift Optimizer},
  author={Degen Serenade},
  url={https://github.com/degenwithheart/chimera-m},
  year={2026},
  version={1.0.0}
}
```

---

## Non-Financial Support

- **Benchmark** on your hardware, file issues with results
- **Test** edge cases, report memory usage and convergence
- **Examples** for popular models (LLaMA, Phi, Qwen)
- **Docs** improvements and clarifications

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Thank you for using CHIMERA-M!**
