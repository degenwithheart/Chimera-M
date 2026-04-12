#!/usr/bin/env python3
"""
CHIMERA-M Training Pipeline

Drop-in training script with automatic hardware detection, model/dataset 
auto-formatting, and autonomous gearshift optimization.

Usage:
    # Basic usage - place model in Models/ and dataset in Datasets/
    python train.py
    
    # With options
    python train.py --epochs 50 --lr 2e-4 --gear auto
    
    # Resume from checkpoint
    python train.py --resume ./Output/checkpoint_epoch3.pt

Arguments:
    --epochs: Training epochs (default: 10)
    --lr: Learning rate (default: 3e-4)
    --batch-size: Batch size (default: 1)
    --gear: Compression level 1-5, or 'auto' (default: auto)
    --models-dir: Model directory (default: ./Models)
    --datasets-dir: Dataset directory (default: ./Datasets)
    --output-dir: Output directory (default: ./Output)

Gear Levels:
    1: BF16 weights, FP8 optimizer - 5× compression
    2: Ternary weights, Count-Min Sketch - 10× compression
    3: Ternary + Sparse gradients - 20× compression
    4: Level 3 + SSD offloading - 40× compression
    5: MEZO + Minimal state + SSD - 50× compression
"""

import sys
from chimera_m import main

if __name__ == '__main__':
    sys.exit(main())
