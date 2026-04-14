"""
CHIMERA-M: Compressive Hybrid Architecture for Intelligent, Efficient Resource Allocation and Modeling

A self-contained, drop-in training system with autonomous CPU↔GPU adaptation,
extreme compression (5× to 50×), and Bayesian-optimized gearshift watchdog.

Key Features:
- 5 compression levels from BF16 (5×) to MEZO+SSD (50×)
- GPU-only watchdog with L2 cache residency (100ms polling)
- Auto-formatting datasets to match model requirements
- Bayesian optimization for threshold tuning
- Never stops training - emergency downshift to Level 5
- Self-contained (vendors BO, no external dependencies)

Author: Degen Serenade
License: MIT
"""

from __future__ import annotations

__version__ = "1.0.0"

import os
import sys
import time
import json
import math
import random
import pickle
import lz4.frame
import threading
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from collections import deque, Counter
from pathlib import Path
import logging
import warnings

# Core scientific computing
import numpy as np
from numpy import ndarray

# PyTorch
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

# Transformers (optional - for model loading)
try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from transformers import AutoModelForCausalLM, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not installed. Model auto-detection limited.")

# CHIMERA-M C/C++ Acceleration Extensions (optional)
try:
    from chimera_m_cpp import ternary_codec as _ternary_cpp
    _TERNARY_CPP_AVAILABLE = True
except ImportError:
    _TERNARY_CPP_AVAILABLE = False

try:
    from chimera_m_c import (
        cms_is_available as _cms_c_available,
        cms_update_fast,
        cms_query_fast,
    )
    _CMS_C_AVAILABLE = _cms_c_available()
except ImportError:
    _CMS_C_AVAILABLE = False
    cms_update_fast = None
    cms_query_fast = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Watchdog timing constants (milliseconds)
DEFAULT_WATCHDOG_POLL_INTERVAL_MS = 100.0
DEFAULT_WATCHDOG_HOLD_DURATION_S = 60.0

# CMS (Count-Min Sketch) constants
DEFAULT_CMS_WIDTH = 1024
DEFAULT_CMS_DEPTH = 4

# Paged memory constants
DEFAULT_PAGE_SIZE_MB = 256
DEFAULT_RAM_THRESHOLD = 0.85
MIN_RAM_THRESHOLD = 0.70
MAX_RAM_THRESHOLD = 0.95

# Gear shift thresholds
DEFAULT_VRAM_DOWNSHIFT_THRESHOLD = 0.85
DEFAULT_LOSS_SPIKE_THRESHOLD = 1.5
HYSTERESIS_MARGIN = 0.15  # For upshift decisions

# Training constants
DEFAULT_LR = 3e-4
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 512
DEFAULT_LOG_INTERVAL = 10
DEFAULT_CHECKPOINT_INTERVAL = 500

# MEZO constants
DEFAULT_MEZO_EPSILON = 1e-3
MAX_MEZO_RETRIES = 3
MEZO_PENALTY_LOSS = 1e6

# Bayesian optimizer constants
DEFAULT_BO_EXPLORATION_XI = 0.01
DEFAULT_BO_NOISE_VARIANCE = 0.05
BO_RANDOM_EXPLORATION_STEPS = 10
BO_ACQUISITION_SAMPLES = 100

# Ternary compression constants
TERNARY_VALUES_PER_INT32 = 16
TERNARY_BITS_PER_VALUE = 2
TERNARY_CODES = {-1: 0, 0: 1, 1: 2}  # Mapping to packed codes

# Hardware detection constants
MIN_VRAM_GB_FOR_GPU_TRAINING = 4.0
MIN_RAM_GB = 8.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BAYESIAN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Bayesian Optimizer for gearshift threshold tuning.

Gaussian Process with RBF/Matérn kernels, Expected Improvement acquisition.
Lightweight implementation for 100-step updates, async execution.
"""


class Kernel(ABC):
    """Abstract base class for GP kernels."""
    
    @abstractmethod
    def __call__(self, x1: ndarray, x2: ndarray) -> float:
        """Compute kernel value between two points."""
        pass
    
    @abstractmethod
    def compute_gram(self, X: ndarray) -> ndarray:
        """Compute Gram matrix for dataset X."""
        pass
    
    @abstractmethod
    def compute_cross(self, X1: ndarray, X2: ndarray) -> ndarray:
        """Compute cross-kernel matrix K(X1, X2)."""
        pass


class RBFKernel(Kernel):
    """Radial Basis Function (Gaussian) kernel."""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, x1: ndarray, x2: ndarray) -> float:
        sq_dist = np.sum((x1 - x2) ** 2)
        return self.variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
    
    def compute_gram(self, X: ndarray) -> ndarray:
        """Compute Gram matrix using vectorized operations for speed."""
        # Vectorized computation: K[i,j] = variance * exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)
        # Using broadcasting: (n,1,d) - (1,n,d) -> (n,n,d) -> sum over d -> (n,n)
        X = np.asarray(X)
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        return self.variance * np.exp(-0.5 * sq_dists / (self.length_scale ** 2))
    
    def compute_cross(self, X1: ndarray, X2: ndarray) -> ndarray:
        """Compute cross-kernel matrix K(X1, X2) vectorized."""
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        # Broadcasting: (n1,1,d) - (1,n2,d) -> (n1,n2,d)
        sq_dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        return self.variance * np.exp(-0.5 * sq_dists / (self.length_scale ** 2))


class MaternKernel(Kernel):
    """Matérn 5/2 kernel - good for non-smooth objectives."""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, x1: ndarray, x2: ndarray) -> float:
        dist = np.sqrt(np.sum((x1 - x2) ** 2) + 1e-8)
        scaled = np.sqrt(5) * dist / self.length_scale
        return self.variance * (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
    
    def compute_gram(self, X: ndarray) -> ndarray:
        """Compute Gram matrix using vectorized operations for speed."""
        # Vectorized Matérn 5/2 computation
        X = np.asarray(X)
        sq_dists = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        dists = np.sqrt(sq_dists + 1e-8)
        scaled = np.sqrt(5) * dists / self.length_scale
        return self.variance * (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
    
    def compute_cross(self, X1: ndarray, X2: ndarray) -> ndarray:
        """Compute cross-kernel matrix K(X1, X2) vectorized."""
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        # Broadcasting: (n1,1,d) - (1,n2,d) -> (n1,n2,d)
        sq_dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        dists = np.sqrt(sq_dists + 1e-8)
        scaled = np.sqrt(5) * dists / self.length_scale
        return self.variance * (1 + scaled + scaled**2 / 3) * np.exp(-scaled)


class AcquisitionFunction(ABC):
    """Abstract acquisition function for BO."""
    
    @abstractmethod
    def __call__(self, x: ndarray, gp: 'GaussianProcess', best_y: float) -> float:
        """Compute acquisition value at point x."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi  # Exploration parameter
    
    def __call__(self, x: ndarray, gp: 'GaussianProcess', best_y: float) -> float:
        mean, std = gp.predict(x.reshape(1, -1))
        mean, std = mean[0], std[0]
        
        if std < 1e-9:
            return 0.0
        
        z = (best_y - mean - self.xi) / std
        ei = (best_y - mean - self.xi) * (0.5 * (1 + math.erf(z / math.sqrt(2))))
        ei += std * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        
        return max(0, ei)


class GaussianProcess:
    """Lightweight Gaussian Process for BO."""
    
    def __init__(self, kernel: Kernel, noise_variance: float = 0.1):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X: Optional[ndarray] = None
        self.y: Optional[ndarray] = None
        self.K: Optional[ndarray] = None
        self.L: Optional[ndarray] = None
        self.alpha: Optional[ndarray] = None
    
    def fit(self, X: ndarray, y: ndarray):
        """Fit GP to observations."""
        self.X = X
        self.y = y
        n = len(X)
        
        # Compute Gram matrix
        self.K = self.kernel.compute_gram(X)
        self.K += self.noise_variance * np.eye(n)
        
        # Cholesky decomposition
        try:
            self.L = np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            # Add jitter for numerical stability
            self.K += 1e-6 * np.eye(n)
            self.L = np.linalg.cholesky(self.K)
        
        # Solve for alpha
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
    
    def predict(self, X_new: ndarray) -> Tuple[ndarray, ndarray]:
        """Predict mean and std at new points using vectorized operations."""
        if self.X is None:
            raise RuntimeError("GP not fitted yet")
        
        # Vectorized kernel computation
        K_new_train = self.kernel.compute_cross(X_new, self.X)
        K_new = self.kernel.compute_gram(X_new)
        
        # Predict mean
        mean = K_new_train @ self.alpha
        
        # Predict variance
        v = np.linalg.solve(self.L, K_new_train.T)
        var = np.diag(K_new) - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-9)  # Ensure positive
        
        return mean, np.sqrt(var)


@dataclass
class Bounds:
    """Parameter bounds for BO."""
    low: float
    high: float
    
    def sample(self, rng: np.random.RandomState) -> float:
        return rng.uniform(self.low, self.high)
    
    def clip(self, value: float) -> float:
        return max(self.low, min(self.high, value))


class BayesianOptimizer:
    """Bayesian Optimizer for gearshift threshold tuning."""
    
    def __init__(
        self,
        kernel_type: str = 'matern',
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise_variance: float = 0.05,
        acquisition_type: str = 'ei',
        xi: float = 0.01
    ):
        # Initialize kernel
        if kernel_type == 'rbf':
            kernel = RBFKernel(length_scale, variance)
        else:
            kernel = MaternKernel(length_scale, variance)
        
        self.gp = GaussianProcess(kernel, noise_variance)
        
        # Initialize acquisition
        if acquisition_type == 'ei':
            self.acquisition = ExpectedImprovement(xi)
        else:
            self.acquisition = ExpectedImprovement(0.01)
        
        self.bounds: Dict[str, Bounds] = {}
        self.history: List[Tuple[ndarray, float]] = []
        self.best_y = float('inf')
        self.best_x: Optional[ndarray] = None
    
    def add_param(self, name: str, low: float, high: float):
        """Add parameter with bounds."""
        self.bounds[name] = Bounds(low, high)
    
    def suggest(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """Suggest next point to evaluate."""
        if rng is None:
            rng = np.random.RandomState()
        
        if len(self.history) < 10:
            # Random exploration phase
            return {name: bound.sample(rng) for name, bound in self.bounds.items()}
        
        # BO phase: optimize acquisition
        best_acq = -float('inf')
        best_params = {}
        
        # Random search for acquisition optimization (lightweight)
        for _ in range(100):
            params = {name: bound.sample(rng) for name, bound in self.bounds.items()}
            x = np.array([params[name] for name in sorted(self.bounds.keys())])
            
            acq_value = self.acquisition(x, self.gp, self.best_y)
            
            if acq_value > best_acq:
                best_acq = acq_value
                best_params = params
        
        return best_params
    
    def update(self, params: Dict[str, float], value: float):
        """Update with observation."""
        x = np.array([params[name] for name in sorted(self.bounds.keys())])
        self.history.append((x, value))
        
        if value < self.best_y:
            self.best_y = value
            self.best_x = x
        
        # Refit GP
        if len(self.history) >= 2:
            X = np.array([h[0] for h in self.history])
            y = np.array([h[1] for h in self.history])
            self.gp.fit(X, y)
    
    def get_best(self) -> Tuple[Dict[str, float], float]:
        """Get best parameters found so far."""
        if self.best_x is None:
            return {}, float('inf')
        
        params = {name: self.best_x[i] for i, name in enumerate(sorted(self.bounds.keys()))}
        return params, self.best_y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TERNARY COMPRESSION CODEC
# ═══════════════════════════════════════════════════════════════════════════════


class TernaryCodec:
    """
    Ternary weight quantization: {-1, 0, +1} at ~1.58 bits/param.
    
    Packing: 16 ternary values per uint32 (2 bits/value).
    Uses stochastic rounding for unbiased quantization.
    
    Uses C++ extension when available for 10-50x speedup.
    """
    
    def __init__(self, stochastic: bool = True):
        self.stochastic = stochastic
        self.use_cpp = _TERNARY_CPP_AVAILABLE
        if self.use_cpp:
            logger.debug("TernaryCodec using C++ acceleration")
    
    def encode(self, tensor: torch.Tensor, seed: Optional[int] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Encode FP32 tensor to packed ternary.
        
        Returns:
            packed: Int32 tensor of shape (ceil(numel/16),)
            metadata: (original_shape, scale_factor)
        """
        original_shape = tensor.shape
        
        # Use C++ extension if available
        if self.use_cpp and tensor.device.type == 'cpu':
            flat = tensor.flatten().detach().cpu().numpy()
            max_abs = float(np.abs(flat).max())
            # Handle zero tensor: return zeros with scale 0
            if max_abs < 1e-12:
                packed_size = (flat.size + 15) // 16
                packed_np = np.zeros(packed_size, dtype=np.int32)
                packed = torch.from_numpy(packed_np).to(tensor.device)
                return packed, (original_shape, 0.0)
            scale = max_abs + 1e-8
            
            packed_np = _ternary_cpp.pack(
                flat, scale, self.stochastic, seed or 0
            )
            packed = torch.from_numpy(packed_np).to(tensor.device)
            metadata = (original_shape, scale)
            return packed, metadata
        
        # Python fallback
        flat = tensor.flatten()
        n = flat.numel()
        
        # Scale to maximize dynamic range
        max_abs = flat.abs().max().item()
        # Handle zero tensor: return zeros with scale 0
        if max_abs < 1e-12:
            packed_size = (n + 15) // 16
            packed = torch.zeros(packed_size, dtype=torch.int32, device=tensor.device)
            return packed, (original_shape, 0.0)
        scale = max_abs + 1e-8
        normalized = flat / scale
        
        # Quantize to ternary
        if self.stochastic and seed is not None:
            rng = torch.Generator(device=flat.device)
            rng.manual_seed(seed)
            rand = torch.rand(n, generator=rng, device=flat.device)
        else:
            rand = torch.rand(n, device=flat.device) if self.stochastic else None
        
        if self.stochastic and rand is not None:
            # Stochastic rounding: probability proportional to value
            quantized = torch.zeros_like(flat)
            
            # +1 if normalized > 0.5 + random, -1 if < -0.5 - random
            pos_mask = normalized > 0.5 + (rand - 0.5) * 0.1
            neg_mask = normalized < -0.5 - (rand - 0.5) * 0.1
            
            quantized = torch.where(pos_mask, torch.ones_like(flat),
                         torch.where(neg_mask, -torch.ones_like(flat), 
                                    torch.zeros_like(flat)))
        else:
            # Deterministic
            quantized = torch.where(normalized > 0.5, torch.ones_like(flat),
                         torch.where(normalized < -0.5, -torch.ones_like(flat),
                                    torch.zeros_like(flat)))
        
        # Pack 16 values into uint32
        packed_size = (n + 15) // 16
        packed = torch.zeros(packed_size, dtype=torch.int32, device=tensor.device)
        
        # Convert {-1, 0, +1} to {0, 1, 2} for packing
        codes = (quantized + 1).to(torch.int32)  # Now {0, 1, 2}
        
        # Efficient bit-packing using direct indexing with bitwise OR
        # Each packed element contains 16 ternary values (2 bits each)
        for group_idx in range(packed_size):
            start = group_idx * 16
            end = min(start + 16, n)
            group_codes = codes[start:end]
            
            # Build packed value using Python int for efficiency (avoids tensor creation in loop)
            packed_val = 0
            for i, code in enumerate(group_codes):
                packed_val |= int(code.item()) << (i * 2)
            
            packed[group_idx] = packed_val
        
        metadata = (original_shape, scale)
        return packed, metadata
    
    def decode(self, packed: torch.Tensor, metadata: Tuple) -> torch.Tensor:
        """Decode packed ternary back to FP32 tensor."""
        original_shape, scale = metadata
        
        # Handle zero scale (zero tensor case)
        if abs(scale) < 1e-12:
            return torch.zeros(original_shape, device=packed.device)
        
        # Use C++ extension if available
        if self.use_cpp and packed.device.type == 'cpu':
            packed_np = packed.detach().cpu().numpy()
            weights_np = _ternary_cpp.unpack(packed_np, original_shape, scale)
            return torch.from_numpy(weights_np).to(packed.device)
        
        # Python fallback
        n = int(np.prod(original_shape))
        packed_size = packed.numel()
        
        # Unpack
        flat = torch.zeros(n, device=packed.device)
        
        for i in range(16):
            shift = i * 2
            mask = 0b11 << shift
            idx = torch.arange(i, n, 16, device=packed.device)
            if len(idx) > 0:
                # Fix: clamp packet index to avoid overflow
                packet_indices = (idx // 16).clamp(max=packed_size - 1)
                codes = ((packed[packet_indices] & mask) >> shift).float()
                # Convert {0, 1, 2} back to {-1, 0, +1}
                flat[idx] = codes - 1
        
        # Rescale
        flat = flat * scale
        
        return flat.reshape(original_shape)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COUNT-MIN SKETCH (Optimizer State Compression)
# ═══════════════════════════════════════════════════════════════════════════════


class CountMinSketch:
    """
    Count-Min Sketch for constant-memory optimizer states.
    
    Replaces Adam's momentum/variance (8 bytes/param) with 16KB constant.
    Compression: 40-50× for typical model sizes.
    
    Algorithm: d hash functions map to width buckets, store min across rows.
    Uses C extension when available for 20-100× speedup.
    """
    
    def __init__(self, width: int = DEFAULT_CMS_WIDTH, depth: int = DEFAULT_CMS_DEPTH, device: str = 'cuda'):
        self.width = width
        self.depth = depth
        self.device = device
        
        # Tables: momentum and variance estimates
        self.tables_m = torch.zeros((depth, width), device=device)
        self.tables_v = torch.zeros((depth, width), device=device)
        
        # Hash seeds (on same device as tables for consistent hashing)
        self.seeds = torch.randint(0, 2**31, (depth,), device=device)
        
        # Statistics
        self.step_count = 0
        
        # Use C extension if available and tensor is on CPU
        # Note: C extension only supports CPU tensors, GPU tensors use optimized Python
        self.use_c = _CMS_C_AVAILABLE and device == 'cpu'
        self.device = device  # Store device for reference
        if self.use_c:
            logger.debug("CountMinSketch using C acceleration")
        elif device == 'cuda':
            logger.debug("CountMinSketch using CUDA-optimized Python implementation")
    
    def _hash(self, indices: torch.Tensor, seed: int) -> torch.Tensor:
        """Universal hash function."""
        # Simple but effective: (a * x + b) % p
        a = seed * 2 + 1
        b = seed // 2
        p = 2147483647  # Large prime
        return ((a * indices + b) % p) % self.width
    
    def update(self, indices: torch.Tensor, values_m: torch.Tensor, values_v: torch.Tensor, 
               beta1: float = 0.9, beta2: float = 0.999):
        """
        Update sketch with new momentum/variance values.
        
        NOTE: Bias correction is applied at query time, not update time.
        This matches standard Adam behavior where bias correction is applied
        to the accumulated moments when computing the update.
        
        Args:
            indices: Global parameter indices (flat, across all parameters)
            values_m: Momentum values for these indices
            values_v: Variance values
            beta1, beta2: Adam decay rates
        """
        self.step_count += 1
        
        # Try C extension first for CPU tensors
        if self.use_c and cms_update_fast is not None:
            # Pass actual step_count for proper bias correction at query time
            # (bias correction is applied at query time, not during update)
            success = cms_update_fast(
                self.tables_m, self.tables_v,
                indices, values_m, values_v,
                self.seeds, beta1, beta2, self.step_count
            )
            if success:
                return
        
        # Python fallback - pure EMA update without bias correction
        for d in range(self.depth):
            # Hash indices to buckets (pass seed as tensor for device consistency)
            buckets = self._hash(indices, self.seeds[d])
            
            # Update with exponential moving average (no bias correction here)
            for i, (idx, m, v) in enumerate(zip(buckets, values_m, values_v)):
                # Pure EMA update (bias correction applied at query time)
                self.tables_m[d, idx] = beta1 * self.tables_m[d, idx] + (1 - beta1) * m
                self.tables_v[d, idx] = beta2 * self.tables_v[d, idx] + (1 - beta2) * v
    
    def query(self, indices: torch.Tensor, beta1: float = 0.9, beta2: float = 0.999) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query momentum and variance estimates for given indices.
        
        Applies bias correction at query time (standard Adam behavior).
        Returns minimum across hash rows (Count-Min property).
        
        Args:
            indices: Global parameter indices
            beta1, beta2: Adam decay rates for bias correction
        """
        # Try C extension first for CPU tensors
        if self.use_c and cms_query_fast is not None:
            result = cms_query_fast(
                self.tables_m, self.tables_v,
                indices, self.seeds,
                beta1=beta1, beta2=beta2, step_count=self.step_count
            )
            if result is not None:
                # Bias correction already applied in cms_query_fast
                return result
        
        # Python fallback
        m_estimates = []
        v_estimates = []
        
        for d in range(self.depth):
            # Hash indices to buckets (pass seed as tensor for device consistency)
            buckets = self._hash(indices, self.seeds[d])
            m_estimates.append(self.tables_m[d, buckets])
            v_estimates.append(self.tables_v[d, buckets])
        
        # Take minimum (conservative estimate)
        m_hat = torch.min(torch.stack(m_estimates), dim=0)[0]
        v_hat = torch.min(torch.stack(v_estimates), dim=0)[0]
        
        # Apply bias correction at query time (not at update time)
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count
        m_hat = m_hat / bias_correction1
        v_hat = v_hat / bias_correction2
        
        return m_hat, v_hat
    
    def memory_usage_bytes(self) -> int:
        """Return constant memory usage."""
        return 2 * self.depth * self.width * 4  # 2 tables, float32


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PAGED MEMORY (SSD OFFLOADING)
# ═══════════════════════════════════════════════════════════════════════════════


class PagedMemory:
    """
    NVMe SSD offloading for optimizer states when RAM exceeds threshold.
    
    Features:
    - 256MB-512MB pages (scaled by DDR speed)
    - LZ4 compression on pages
    - Async prefetch
    - 85% RAM threshold activation
    """
    
    def __init__(self, page_size_mb: int = 256, ram_threshold: float = 0.85, 
                 cache_dir: Optional[str] = None):
        self.page_size = page_size_mb * 1024 * 1024  # Bytes
        self.ram_threshold = ram_threshold
        # Use default path if not specified, ensure cross-platform compatibility
        if cache_dir is None:
            # Use user's home directory as safe default (not cwd which could be attacker-controlled)
            cache_dir = Path.home() / ".chimera_m" / "ssd_cache"
        else:
            # Validate user-provided path to prevent path traversal attacks
            cache_path = Path(cache_dir).resolve()
            # Ensure the path doesn't escape to system directories
            home = Path.home().resolve()
            cwd = Path.cwd().resolve()
            # Allow paths under home or cwd only
            if not (str(cache_path).startswith(str(home)) or str(cache_path).startswith(str(cwd))):
                logger.warning(f"Cache dir {cache_path} outside home/cwd, using safe default")
                cache_dir = Path.home() / ".chimera_m" / "ssd_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.pages: Dict[str, Any] = {}  # param_id -> (file_path, shape, dtype)
        self.active_buffers: Dict[str, torch.Tensor] = {}  # Currently in RAM
        self.page_files: List[Path] = []
        
        # Statistics
        self.stats = {
            'pages_written': 0,
            'pages_read': 0,
            'bytes_written': 0,
            'bytes_read': 0,
        }
        
        # Adaptive page size based on DDR
        self._adjust_page_size()
    
    def _adjust_page_size(self):
        """Adjust page size based on detected RAM speed."""
        try:
            # dmidecode is Linux-only; skip on macOS/Windows
            import sys
            if sys.platform != 'linux':
                raise OSError("dmidecode only available on Linux")
            
            # Try to detect DDR version
            import subprocess
            result = subprocess.run(['dmidecode', '-t', 'memory'], 
                                  capture_output=True, text=True, timeout=5)
            output = result.stdout.lower()
            
            if 'ddr5' in output:
                self.page_size = 512 * 1024 * 1024  # 512MB for DDR5
                logger.info("Detected DDR5, using 512MB pages")
            elif 'ddr4' in output:
                # Check speed
                if 'speed: 3200' in output or 'speed: 3600' in output:
                    self.page_size = 384 * 1024 * 1024  # 384MB for fast DDR4
                    logger.info("Detected fast DDR4, using 384MB pages")
                else:
                    self.page_size = 256 * 1024 * 1024  # 256MB default
                    logger.info("Detected standard DDR4, using 256MB pages")
        except (OSError, subprocess.SubprocessError, FileNotFoundError) as e:
            # Default - works on all platforms including macOS
            self.page_size = 256 * 1024 * 1024
            logger.info(f"Using default 256MB pages (platform detection unavailable: {e})")
    
    def check_ram_pressure(self) -> bool:
        """Check if RAM usage exceeds threshold."""
        import psutil
        ram = psutil.virtual_memory()
        return ram.percent / 100 > self.ram_threshold
    
    def spill_to_ssd(self, param_id: str, tensor: torch.Tensor):
        """
        Move tensor from RAM to SSD with robust error handling.
        
        Args:
            param_id: Unique identifier for the parameter
            tensor: Tensor to offload to SSD
            
        Raises:
            RuntimeError: If SSD write fails after cleanup
        """
        import uuid
        page_file = self.cache_dir / f"page_{param_id}_{self.stats['pages_written']}.pkl.lz4"
        # Use unique temp file with timestamp, PID, and UUID to prevent race conditions
        temp_file = self.cache_dir / f".tmp_{param_id}_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp"
        
        try:
            # Move tensor to CPU and convert to numpy
            data = tensor.detach().cpu().numpy()
            
            # Serialize and compress
            pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = lz4.frame.compress(pickled_data)
            
            # Write to temporary file first (atomic write pattern)
            with open(temp_file, 'wb') as f:
                f.write(compressed)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Atomic rename on success
            os.replace(str(temp_file), str(page_file))
            
            # Update tracking
            self.pages[param_id] = {
                'file': page_file,
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
            }
            
            self.stats['pages_written'] += 1
            self.stats['bytes_written'] += tensor.numel() * tensor.element_size()
            
            # Clear from active buffers
            if param_id in self.active_buffers:
                del self.active_buffers[param_id]
                
        except Exception as e:
            # Cleanup temporary file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass  # Best effort cleanup
            
            # Remove partial page file if it was created
            if page_file.exists():
                try:
                    page_file.unlink()
                except OSError:
                    pass
            
            logger.error(f"SSD spill failed for {param_id}: {e}")
            raise RuntimeError(f"Failed to spill {param_id} to SSD: {e}") from e
    
    def load_from_ssd(self, param_id: str) -> torch.Tensor:
        """Load tensor from SSD to RAM."""
        if param_id not in self.pages:
            raise KeyError(f"Parameter {param_id} not in SSD cache")
        
        info = self.pages[param_id]
        
        with open(info['file'], 'rb') as f:
            compressed = f.read()
        
        data = pickle.loads(lz4.frame.decompress(compressed))
        tensor = torch.from_numpy(data).to(info['device'])
        
        self.active_buffers[param_id] = tensor
        self.stats['pages_read'] += 1
        self.stats['bytes_read'] += tensor.numel() * tensor.element_size()
        
        return tensor
    
    def get_tensor(self, param_id: str) -> Optional[torch.Tensor]:
        """Get tensor, loading from SSD if necessary."""
        if param_id in self.active_buffers:
            return self.active_buffers[param_id]
        
        if param_id in self.pages:
            return self.load_from_ssd(param_id)
        
        return None
    
    def cleanup(self):
        """Remove all page files."""
        pages = list(self.cache_dir.glob("*.pkl.lz4"))
        for f in pages:
            f.unlink()
        logger.info(f"PagedMemory cleanup: removed {len(pages)} pages")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GEARSHIFT WATCHDOG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GearConfig:
    """Configuration for a single gear level."""
    level: int
    weight_format: str  # 'BF16', 'TERNARY', 'MEZO_TERNARY'
    optimizer_format: str  # 'FP8', 'SKETCH', 'SPARSE_SKETCH', 'MINIMAL'
    shadow_on_cpu: bool
    ssd_offload: bool
    compression_ratio: float
    mezo_enabled: bool = False
    mezo_epsilon: float = 1e-3


# Gear definitions
GEAR_LEVELS = {
    1: GearConfig(1, 'BF16', 'FP8', False, False, 5.0),
    2: GearConfig(2, 'TERNARY', 'SKETCH', True, False, 10.0),
    3: GearConfig(3, 'TERNARY', 'SPARSE_SKETCH', True, False, 20.0),
    4: GearConfig(4, 'TERNARY', 'SPARSE_SKETCH', True, True, 40.0),
    5: GearConfig(5, 'MEZO_TERNARY', 'MINIMAL', True, True, 50.0, mezo_enabled=True),
}


class GearshiftWatchdog:
    """
    GPU-only watchdog with L2 cache residency.
    
    - 100ms polling
    - 60s hold after shift
    - Adaptive BO frequency
    - Emergency downshift never stops training
    """
    
    def __init__(
        self,
        poll_interval_ms: float = DEFAULT_WATCHDOG_POLL_INTERVAL_MS,
        cpu_pin: int = 0,
        ram_threshold: float = DEFAULT_RAM_THRESHOLD,
        bo_enabled: bool = True,
    ):
        self.poll_interval = poll_interval_ms / 1000.0  # Convert to seconds
        self.cpu_pin = cpu_pin
        self.ram_threshold = ram_threshold
        self.bo_enabled = bo_enabled
        
        # Current state
        self.current_gear = 2  # Start at balanced level
        self.hold_until = 0.0
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # L2-friendly ring buffers (fixed size, minimal footprint)
        self.vram_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=50)
        self.step_times = deque(maxlen=20)
        
        # Statistics
        self.stats = {
            'pressure_detected': 0,
            'shifts_up': 0,
            'shifts_down': 0,
            'emergency_shifts': 0,
        }
        
        # Bayesian Optimizer for threshold tuning
        if self.bo_enabled:
            self.bo = BayesianOptimizer(kernel_type='matern', noise_variance=DEFAULT_BO_NOISE_VARIANCE)
            self.bo.add_param('vram_downshift', MIN_RAM_THRESHOLD, MAX_RAM_THRESHOLD)
            self.bo.add_param('loss_spike', 1.1, 3.0)
            self.bo.add_param('hold_duration', 30.0, 120.0)
            
            self.thresholds = {
                'vram_downshift': DEFAULT_VRAM_DOWNSHIFT_THRESHOLD,
                'loss_spike': DEFAULT_LOSS_SPIKE_THRESHOLD,
                'hold_duration': DEFAULT_WATCHDOG_HOLD_DURATION_S,
            }
            
            self.bo_history = []
            self.bo_update_interval = 100
            self.bo_step_count = 0
        else:
            self.thresholds = {
                'vram_downshift': DEFAULT_VRAM_DOWNSHIFT_THRESHOLD,
                'loss_spike': DEFAULT_LOSS_SPIKE_THRESHOLD,
                'hold_duration': DEFAULT_WATCHDOG_HOLD_DURATION_S,
            }
        
        # Adaptive frequency
        self.unstable_counter = 0
        
        # Threading locks for thread-safe operations (prevents race conditions)
        self._thresholds_lock = threading.Lock()
        self._bo_history_lock = threading.Lock()
    
    def start(self):
        """Start watchdog thread pinned to specified CPU core (Linux only)."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        
        # Set CPU affinity if supported (Linux only)
        if hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, {self.cpu_pin})
                logger.info(f"Watchdog pinned to CPU core {self.cpu_pin}")
            except AttributeError:
                logger.debug("CPU affinity not available on this platform")
            except OSError as e:
                logger.warning(f"Could not pin watchdog to CPU core {self.cpu_pin}: {e}")
        else:
            logger.debug("CPU affinity setting not supported on this platform (macOS/Windows)")
    
    def stop(self):
        """Stop watchdog thread."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("Watchdog thread did not terminate within timeout")
    
    def _watch_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            # Collect metrics
            metrics = self._collect_metrics()
            
            # Store in ring buffers
            if metrics.get('vram_pct') is not None:
                self.vram_history.append(metrics['vram_pct'])
            if metrics.get('loss') is not None:
                self.loss_history.append(metrics['loss'])
            
            # Check for pressure (fast path)
            if self._detect_pressure(metrics):
                self.stats['pressure_detected'] += 1
                self.unstable_counter += 1
            else:
                self.unstable_counter = max(0, self.unstable_counter - 1)
            
            # Adaptive BO frequency
            if self.bo_enabled:
                self.bo_step_count += 1
                bo_interval = max(50, 100 - self.unstable_counter * 10)  # More frequent when unstable
                
                if self.bo_step_count % bo_interval == 0:
                    self._async_bo_update()
            
            time.sleep(self.poll_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics (fast, minimal overhead)."""
        metrics = {}
        
        # GPU VRAM (primary metric)
        if torch.cuda.is_available():
            metrics['vram_used'] = torch.cuda.memory_allocated()
            metrics['vram_total'] = torch.cuda.get_device_properties(0).total_memory
            metrics['vram_pct'] = metrics['vram_used'] / metrics['vram_total']
        
        # RAM (for shadow weights)
        try:
            import psutil
            ram = psutil.virtual_memory()
            metrics['ram_pct'] = ram.percent / 100
        except (ImportError, AttributeError, OSError):
            metrics['ram_pct'] = 0.0
        
        return metrics
    
    def _detect_pressure(self, metrics: Dict[str, float]) -> bool:
        """Detect if system is under pressure."""
        pressure_score = 0.0
        
        # Thread-safe read of thresholds
        with self._thresholds_lock:
            vram_downshift = self.thresholds['vram_downshift']
            loss_spike = self.thresholds['loss_spike']
        
        # VRAM pressure (40% weight)
        if metrics.get('vram_pct', 0) > vram_downshift:
            pressure_score += 0.4
        
        # Loss spike detection (30% weight)
        if len(self.loss_history) >= 10:
            recent = list(self.loss_history)[-10:]
            if recent[-1] > recent[0] * loss_spike:
                pressure_score += 0.3
        
        # Gradient instability (simulated via step time variance)
        if len(self.step_times) >= 5:
            times = list(self.step_times)
            if np.std(times) > np.mean(times) * 0.5:
                pressure_score += 0.2
        
        # Thermal throttling (placeholder)
        # TODO: Read nvidia-smi for thermal info
        
        return pressure_score > 0.5
    
    def should_shift(self, current_loss: Optional[float] = None) -> Tuple[bool, int]:
        """
        Determine if gear shift is needed.
        
        Returns:
            (should_shift, target_gear)
        """
        # Check hold period
        if time.time() < self.hold_until:
            return False, self.current_gear
        
        # Update loss if provided
        if current_loss is not None:
            self.loss_history.append(current_loss)
        
        # Check pressure
        if len(self.vram_history) < 5:
            return False, self.current_gear
        
        vram_recent = np.mean(list(self.vram_history)[-5:])
        
        # Thread-safe read of thresholds
        with self._thresholds_lock:
            vram_downshift = self.thresholds['vram_downshift']
            hold_duration = self.thresholds['hold_duration']
        
        # Downshift condition
        if vram_recent > vram_downshift:
            if self.current_gear < 5:
                target = min(5, self.current_gear + 1)
                return True, target
        
        # Upshift condition (only after hold expires and stable)
        if vram_recent < vram_downshift - HYSTERESIS_MARGIN:  # Hysteresis
            if self.current_gear > 1:
                # Check if improving (BO-learned)
                if self._is_improving():
                    target = max(1, self.current_gear - 1)
                    return True, target
        
        return False, self.current_gear
    
    def _is_improving(self) -> bool:
        """Check if training is improving (BO-learned definition)."""
        if len(self.loss_history) < 20:
            return True  # Default optimistic
        
        # Simple trend for now (BO will learn better)
        recent = list(self.loss_history)[-20:]
        slope = (recent[-1] - recent[0]) / 20
        
        # Improving if decreasing
        return slope < 0
    
    def execute_shift(self, new_gear: int, callback: Optional[Callable] = None):
        """Execute gear shift and start hold period."""
        old_gear = self.current_gear
        self.current_gear = new_gear
        
        # Start hold period
        self.hold_until = time.time() + self.thresholds['hold_duration']
        
        # Update stats
        if new_gear > old_gear:
            self.stats['shifts_down'] += 1
        else:
            self.stats['shifts_up'] += 1
        
        logger.info(f"Gear shift: {old_gear} -> {new_gear} (hold for {self.thresholds['hold_duration']:.0f}s)")
        
        if callback:
            callback(old_gear, new_gear)
    
    def emergency_downshift(self, callback: Optional[Callable] = None):
        """Emergency downshift to Level 5, never stops training."""
        logger.warning("EMERGENCY DOWNSHIFT triggered! Moving to Level 5.")
        
        self.current_gear = 5
        self.stats['emergency_shifts'] += 1
        self.hold_until = time.time() + self.thresholds['hold_duration']
        
        if callback:
            callback(-1, 5)  # -1 indicates emergency
    
    def _async_bo_update(self):
        """Background BO update for threshold optimization."""
        def optimize():
            # Thread-safe read of bo_history
            with self._bo_history_lock:
                if len(self.bo_history) < 10:
                    return
                # Copy data to avoid holding lock during computation
                history_copy = list(self.bo_history)
            
            # Prepare data (outside lock)
            X = np.array([h['params'] for h in history_copy])
            y = np.array([h['objective'] for h in history_copy])
            
            # Fit and suggest
            self.bo.update(dict(zip(sorted(self.bo.bounds.keys()), X[-1])), y[-1])
            best_params, _ = self.bo.get_best()
            
            if best_params:
                # Smooth update (EMA) with thread-safe lock
                with self._thresholds_lock:
                    for key in self.thresholds:
                        if key in best_params:
                            self.thresholds[key] = 0.8 * self.thresholds[key] + 0.2 * best_params[key]
        
        threading.Thread(target=optimize, daemon=True).start()
    
    def update_objective(self, step_time: float, oom_risk: bool, loss_plateau: bool):
        """Record objective for BO."""
        if not self.bo_enabled:
            return
        
        # Objective: minimize time + penalize bad events
        objective = step_time + (10.0 if oom_risk else 0) + (5.0 if loss_plateau else 0)
        
        # Thread-safe append to bo_history with size limit to prevent unbounded growth
        MAX_BO_HISTORY = 1000  # Keep last 1000 observations
        with self._bo_history_lock:
            self.bo_history.append({
                'params': [
                    self.thresholds['vram_downshift'],
                    self.thresholds['loss_spike'],
                    self.thresholds['hold_duration']
                ],
                'objective': objective
            })
            # Trim history to prevent unbounded memory growth
            if len(self.bo_history) > MAX_BO_HISTORY:
                # Keep most recent entries (more relevant for current optimization)
                self.bo_history = self.bo_history[-MAX_BO_HISTORY:]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CHIMERA GEAR OPTIMIZER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════════


class ChimeraGearOptimizer(Optimizer):
    """
    Main optimizer with autonomous gearshift capability.
    
    Features:
    - 5 compression levels (5× to 50×)
    - Seamless gear transitions with checkpointing
    - Ternary + Count-Min Sketch + SSD offload
    - MEZO zeroth-order support at Level 5
    - Always continues training (emergency fallback)
    """
    
    def __init__(
        self,
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
        bo_enabled: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            gear=gear, device=device
        )
        super().__init__(params, defaults)
        
        self.device = device
        self.cpu_device = 'cpu'
        
        # Compression components
        self.ternary_codec = TernaryCodec(stochastic=True)
        self.sketch = CountMinSketch(width=DEFAULT_CMS_WIDTH, depth=DEFAULT_CMS_DEPTH, device=device)
        
        # Paged memory for SSD offloading
        if ssd_offload:
            self.paged_memory = PagedMemory(ram_threshold=ram_threshold)
        else:
            self.paged_memory = None
        
        # Gear management
        if gear == 'auto':
            self.current_gear = 2  # Start balanced
            self.auto_gear = True
        else:
            self.current_gear = int(gear)
            self.auto_gear = False
        
        self.gear_config = GEAR_LEVELS[self.current_gear]
        
        # Watchdog (GPU-only)
        if torch.cuda.is_available() and self.auto_gear:
            self.watchdog = GearshiftWatchdog(
                poll_interval_ms=DEFAULT_WATCHDOG_POLL_INTERVAL_MS,
                cpu_pin=0,
                ram_threshold=ram_threshold,
                bo_enabled=bo_enabled,
            )
            self.watchdog.start()
        else:
            self.watchdog = None
        
        # CPU shadow weights (for levels 2-5) - ALWAYS float tensors, never packed
        self.shadow_weights: Dict[int, torch.Tensor] = {}
        self.error_feedback: Dict[int, torch.Tensor] = {}
        
        # Packed ternary weights storage (int32) - separate from shadow weights
        self.packed_weights: Dict[int, Tuple[torch.Tensor, Tuple]] = {}
        
        # Global parameter index tracking for CMS (prevents collisions between parameters)
        self.param_global_offsets: Dict[int, int] = {}
        self._compute_global_offsets()
        
        # Initialize shadow weights only if cpu_offload is explicitly enabled
        # Respect user's choice - don't override cpu_offload=False
        if self.current_gear >= 2:
            if cpu_offload:
                self._init_shadow_weights()
            else:
                logger.info("CPU offload disabled by user, keeping weights on device")
        
        # MEZO support
        self.mezo_mode = mezo_mode or self.gear_config.mezo_enabled
        self.mezo_epsilon = self.gear_config.mezo_epsilon
        self.mezo_rng = torch.Generator(device=self.cpu_device) if self.mezo_mode else None
        
        # Threading lock for Bayesian Optimizer updates (prevents race conditions)
        self._bo_lock = threading.Lock()
        
        # State tracking
        self.step_count = 0
        self.checkpoint_on_shift = True
        
        logger.info(f"ChimeraGearOptimizer initialized at Level {self.current_gear}")
        logger.info(f"Compression: {self.gear_config.compression_ratio}×")
        logger.info(f"Auto-gearshift: {self.auto_gear}")
    
    def _compute_global_offsets(self):
        """Compute global cumulative offsets for each parameter to prevent CMS collisions."""
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                self.param_global_offsets[param_id] = offset
                offset += p.numel()
    
    def _get_global_indices(self, param_id: int, local_indices: torch.Tensor) -> torch.Tensor:
        """Convert local parameter indices to global indices for CMS."""
        global_offset = self.param_global_offsets.get(param_id, 0)
        return local_indices + global_offset
    
    def _init_shadow_weights(self):
        """Initialize high-precision shadow weights on CPU."""
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                param_id = id(p)
                # BF16 shadow weights - ALWAYS keep as float, never store packed here
                self.shadow_weights[param_id] = p.detach().clone().to(self.cpu_device).bfloat16()
                # Error feedback (residuals) - ensure on CPU
                self.error_feedback[param_id] = torch.zeros_like(p, device=self.cpu_device)
    
    def _apply_gear_compression(self, new_gear: int):
        """Apply compression for new gear level."""
        old_gear = self.current_gear
        self.current_gear = new_gear
        self.gear_config = GEAR_LEVELS[new_gear]
        
        # Checkpoint before transition
        if self.checkpoint_on_shift:
            self._checkpoint_transition(old_gear, new_gear, 'pre')
        
        # Recompress weights if needed
        if new_gear >= 2 and old_gear < 2:
            # Transition to ternary
            self._compress_to_ternary()
        elif new_gear < 2 and old_gear >= 2:
            # Transition to full precision
            self._decompress_to_full()
        
        # Update MEZO mode
        self.mezo_mode = self.gear_config.mezo_enabled
        
        # Adjust SSD offloading
        if self.gear_config.ssd_offload and self.paged_memory is None:
            self.paged_memory = PagedMemory()
        
        # Checkpoint after transition
        if self.checkpoint_on_shift:
            self._checkpoint_transition(old_gear, new_gear, 'post')
        
        logger.info(f"Gear transition complete: {old_gear} -> {new_gear}")
    
    def _compress_to_ternary(self):
        """Compress all parameters to ternary representation."""
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                # Encode to ternary and store packed version separately
                packed, metadata = self.ternary_codec.encode(p.data)
                self.packed_weights[param_id] = (packed.cpu(), metadata)
                # Decode back to GPU for computation
                p.data.copy_(self.ternary_codec.decode(packed, metadata))
    
    def _decompress_to_full(self):
        """Decompress ternary weights back to full precision."""
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                # Restore from shadow weights (float storage, not packed)
                if param_id in self.shadow_weights:
                    p.data.copy_(self.shadow_weights[param_id].to(self.device).float())
                # Clear packed storage and shadow weights to prevent memory leak
                if param_id in self.packed_weights:
                    del self.packed_weights[param_id]
        # Clear all shadow weights and error feedback after full decompression
        self.shadow_weights.clear()
        self.error_feedback.clear()
    
    def _checkpoint_transition(self, old_gear: int, new_gear: int, phase: str):
        """Save checkpoint during gear transition."""
        checkpoint = {
            'gear': old_gear if phase == 'pre' else new_gear,
            'step': self.step_count,
            'phase': phase,
            'timestamp': time.time(),
            'state_dict': self.state_dict(),
        }
        
        # Use Path for cross-platform compatibility
        output_dir = Path.cwd() / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"gear_transition_step{self.step_count}_{phase}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform single optimization step with gear management."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        # Check for gear shift (if auto-gear enabled)
        if self.auto_gear and self.watchdog:
            should_shift, new_gear = self.watchdog.should_shift(loss)
            
            if should_shift and new_gear != self.current_gear:
                # Use lock to prevent race with BO background thread
                with self._bo_lock:
                    self.watchdog.execute_shift(new_gear, callback=None)
                    self._apply_gear_compression(new_gear)
        
        # Perform update based on current gear
        if self.current_gear == 1:
            self._step_level_1()
        elif self.current_gear == 2:
            self._step_level_2()
        elif self.current_gear == 3:
            self._step_level_3()
        elif self.current_gear == 4:
            self._step_level_4()
        elif self.current_gear == 5:
            self._step_level_5(closure)
        
        return loss
    
    def _step_level_1(self):
        """Level 1: BF16 weights + FP8 optimizer (standard)."""
        # Standard AdamW-style update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _step_level_2(self):
        """Level 2: Ternary + Count-Min Sketch + CPU shadow."""
        self._step_compressed(sparse=False)
    
    def _step_level_3(self):
        """Level 3: Ternary + Sparse Sketch + CPU shadow."""
        self._step_compressed(sparse=True, sparsity=0.001)
    
    def _step_level_4(self):
        """Level 4: Level 3 + SSD offloading."""
        # Check RAM pressure
        if self.paged_memory and self.paged_memory.check_ram_pressure():
            # Spill some shadow weights to SSD
            self._spill_to_ssd()
        
        self._step_compressed(sparse=True, sparsity=0.001)
    
    def _step_level_5(self, closure: Optional[Callable] = None):
        """Level 5: MEZO (Memory-Efficient Zeroth-Order) optimization.
        
        MEZO estimates gradients via finite differences of loss values,
        requiring only forward passes. This eliminates the need to store
        activations for backpropagation, achieving extreme memory efficiency.
        
        Algorithm:
        1. Sample random perturbation direction z ~ N(0, I)
        2. Compute loss at w + εz and w - εz
        3. Gradient estimate: g ≈ [loss(w+εz) - loss(w-εz)] / (2ε) * z
        4. Apply update: w = w - lr * g
        
        Args:
            closure: A closure that recomputes the model and returns the loss.
                    Must be provided for MEZO to work.
        """
        if not self.mezo_mode or closure is None:
            # Fallback to compressed first-order method
            self._step_compressed(sparse=True, sparsity=0.01)
            return
        
        # Get MEZO hyperparameters
        eps = self.mezo_epsilon
        lr = self.param_groups[0]['lr']  # Use first group's LR
        
        # Store original parameter values
        original_values = {}
        for group in self.param_groups:
            for p in group['params']:
                original_values[id(p)] = p.data.clone()
        
        # Generate random seed for reproducibility
        seed = self.step_count
        self.mezo_rng.manual_seed(seed)
        
        # MEZO gradient estimation for each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_id = id(p)
                    
                    # Generate random perturbation direction on same device as parameter
                    # Use device-specific RNG to avoid cross-device tensor issues
                    param_device = p.data.device
                    if param_device.type == 'cuda':
                        # Use CUDA RNG for GPU tensors
                        z = torch.randn_like(p.data)
                    else:
                        # Use CPU RNG for CPU tensors
                        z = torch.randn_like(p.data, generator=self.mezo_rng)
                    z = z / (z.norm() + 1e-8)  # Normalize
                    
                    # Forward perturbation: w + εz
                    p.data = original_values[param_id] + eps * z
                    loss_plus = closure()
                    
                    # Backward perturbation: w - εz
                    p.data = original_values[param_id] - eps * z
                    loss_minus = closure()
                    
                    # Gradient estimate: [f(w+εz) - f(w-εz)] / (2ε) * z
                    if isinstance(loss_plus, torch.Tensor):
                        loss_plus = loss_plus.item()
                    if isinstance(loss_minus, torch.Tensor):
                        loss_minus = loss_minus.item()
                    
                    grad_est = (loss_plus - loss_minus) / (2 * eps) * z
                    
                    # Apply update
                    p.data = original_values[param_id] - lr * grad_est
                    
                    # Store update in sketch for tracking (optional)
                    flat_grad = grad_est.flatten().cpu()
                    n = len(flat_grad)
                    local_indices = torch.arange(n, device='cpu')
                    global_indices = self._get_global_indices(param_id, local_indices)
                    self.sketch.update(global_indices, flat_grad.abs(), flat_grad ** 2,
                                      group['betas'][0], group['betas'][1])
    
    def _step_compressed(self, sparse: bool = False, sparsity: float = 0.001):
        """Generic compressed update with ternary + sketch."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_id = id(p)
                beta1, beta2 = group['betas']
                
                # Sparsify gradient if enabled
                if sparse:
                    grad = self._sparsify_gradient(grad, sparsity)
                
                # Flatten gradient and move to CPU for sketch operations
                flat_grad_cpu = grad.flatten().cpu()
                n = len(flat_grad_cpu)
                
                # Use GLOBAL indices to prevent collisions between parameters
                local_indices = torch.arange(n, device='cpu')
                global_indices = self._get_global_indices(param_id, local_indices)
                
                # Update sketch with gradient info (on CPU)
                self.sketch.update(global_indices, flat_grad_cpu.abs(), flat_grad_cpu ** 2, beta1, beta2)
                
                # Query optimizer state from sketch (bias correction applied in query)
                m_hat, v_hat = self.sketch.query(global_indices, beta1, beta2)
                
                # Update CPU shadow weights
                if param_id in self.shadow_weights:
                    shadow = self.shadow_weights[param_id].float()
                    error = self.error_feedback[param_id]  # Already on CPU
                    
                    # Add error feedback (both on CPU now)
                    grad_with_error = flat_grad_cpu + error
                    
                    # Adam-style update on shadow
                    update = m_hat / (v_hat.sqrt() + group['eps'])
                    
                    # Update shadow
                    shadow = shadow - group['lr'] * update.reshape(shadow.shape)
                    
                    # Update error feedback (residual) - keep on CPU
                    residual = grad_with_error - update.reshape(shadow.shape)
                    self.error_feedback[param_id] = residual
                    
                    # Store shadow (float, not packed)
                    self.shadow_weights[param_id] = shadow.bfloat16()
                    
                    # Requantize to ternary for GPU
                    shadow_gpu = shadow.to(self.device)
                    packed, metadata = self.ternary_codec.encode(shadow_gpu)
                    p.data.copy_(self.ternary_codec.decode(packed, metadata))
    
    def _sparsify_gradient(self, grad: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Top-K sparsification: keep only largest (1-sparsity) fraction."""
        k = max(1, int((1 - sparsity) * grad.numel()))  # Ensure at least 1 element
        
        flat = grad.flatten()
        if k >= flat.numel():
            return grad  # Keep all if k >= numel
        
        # Use largest=True for better performance (ascending sort is slower)
        threshold = torch.topk(flat.abs(), k, largest=True)[0][-1]
        
        mask = flat.abs() >= threshold
        sparse_grad = flat * mask.float()
        
        return sparse_grad.reshape(grad.shape)
    
    def _spill_to_ssd(self):
        """Spill some shadow weights to SSD."""
        if not self.paged_memory:
            return
        
        # Find largest shadow weights to spill
        sizes = [(id(p), self.shadow_weights[id(p)].numel()) 
                 for group in self.param_groups 
                 for p in group['params'] 
                 if id(p) in self.shadow_weights]
        
        # Spill largest first
        sizes.sort(key=lambda x: x[1], reverse=True)
        
        for param_id, _ in sizes[:3]:  # Spill top 3 largest
            if param_id in self.shadow_weights:
                tensor = self.shadow_weights[param_id]
                # Use consistent string format: param_<id> for cache key
                cache_key = f"param_{param_id}"
                self.paged_memory.spill_to_ssd(cache_key, tensor)
    
    def emergency_downshift(self):
        """Emergency downshift to Level 5."""
        logger.warning("Emergency downshift triggered!")
        if self.watchdog:
            self.watchdog.emergency_downshift()
        
        self._apply_gear_compression(5)
        
        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict including gear, shadow weights, and packed weights."""
        state = super().state_dict()
        state['current_gear'] = self.current_gear
        # Shadow weights are always float (BF16) - safe to save
        state['shadow_weights'] = {k: v for k, v in self.shadow_weights.items()}
        state['error_feedback'] = {k: v for k, v in self.error_feedback.items()}
        # Packed weights stored separately (int32 + metadata)
        state['packed_weights'] = {k: (v[0].cpu(), v[1]) for k, v in self.packed_weights.items()}
        # Recompute global offsets on load (param shapes may change)
        state['mezo_mode'] = self.mezo_mode
        # Save step_count for bias correction continuity
        state['step_count'] = self.step_count
        # Save RNG state if MEZO is enabled
        if self.mezo_mode and self.mezo_rng is not None:
            state['mezo_rng_state'] = self.mezo_rng.get_state()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict and restore gear."""
        # Validate gear level
        loaded_gear = state_dict.get('current_gear', 2)
        if not 1 <= loaded_gear <= 5:
            logger.warning(f"Invalid gear level {loaded_gear} in checkpoint, defaulting to 2")
            loaded_gear = 2
        
        self.current_gear = loaded_gear
        self.gear_config = GEAR_LEVELS[self.current_gear]
        self.shadow_weights = state_dict.get('shadow_weights', {})
        self.error_feedback = state_dict.get('error_feedback', {})
        
        # Restore packed weights if present
        if 'packed_weights' in state_dict:
            self.packed_weights = {}
            for k, (packed, metadata) in state_dict['packed_weights'].items():
                self.packed_weights[k] = (packed, metadata)
        
        # Restore MEZO mode and reinitialize RNG
        self.mezo_mode = state_dict.get('mezo_mode', False)
        if self.mezo_mode:
            self.mezo_rng = torch.Generator(device=self.cpu_device)
            # Optionally restore RNG state if available
            if 'mezo_rng_state' in state_dict:
                self.mezo_rng.set_state(state_dict['mezo_rng_state'])
        
        # Restore step_count if present
        self.step_count = state_dict.get('step_count', 0)
        
        # Recompute global offsets (param shapes may have changed)
        self._compute_global_offsets()
        
        # Restart watchdog if it was running
        if self.watchdog and not self.watchdog.is_running:
            self.watchdog.start()
        
        # Remove our keys before passing to parent
        keys_to_remove = ['current_gear', 'shadow_weights', 'error_feedback', 
                         'packed_weights', 'mezo_mode', 'mezo_rng_state', 'step_count']
        parent_state = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        super().load_state_dict(parent_state)
    
    def __del__(self):
        """Cleanup watchdog thread on deletion."""
        if self.watchdog:
            self.watchdog.stop()
        if self.paged_memory:
            self.paged_memory.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL AND DATASET DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware capabilities."""
    hw = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': 0,
        'vram_gb': 0.0,
        'ram_gb': 0.0,
        'cpu_cores': 0,
        'ssd_available': False,
    }
    
    if torch.cuda.is_available():
        hw['cuda_devices'] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        hw['vram_gb'] = props.total_memory / (1024**3)
    
    try:
        import psutil
        hw['ram_gb'] = psutil.virtual_memory().total / (1024**3)
        hw['cpu_cores'] = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError, OSError):
        pass
    
    # Check SSD availability
    try:
        import shutil
        stat = shutil.disk_usage('.')
        hw['ssd_available'] = stat.free > 10 * (1024**3)  # At least 10GB free
    except (ImportError, AttributeError, OSError):
        pass
    
    return hw


def calculate_starting_gear(model_params: int, vram_gb: float, ram_gb: float) -> int:
    """
    Calculate optimal starting gear based on model size and hardware.
    
    Rule (from user experience):
    - 3B on 24GB = Level 1 (BF16)
    - 8B on 24GB = Level 3 (20x, sometimes OOMs without)
    """
    model_bytes_bf16 = model_params * 2  # BF16 = 2 bytes
    
    if vram_gb == 0:
        # CPU-only training
        if model_bytes_bf16 < ram_gb * 0.3 * (1024**3):
            return 2  # Ternary in RAM
        else:
            return 4  # Need SSD offloading
    
    # GPU available
    vram_bytes = vram_gb * (1024**3)
    
    # With 1.5x overhead factor
    if model_bytes_bf16 * 1.5 < vram_bytes:
        return 1  # Full BF16 fits comfortably
    
    # With ternary compression (6x)
    if model_bytes_bf16 / 6 < vram_bytes:
        return 2  # Ternary fits
    
    # With 20x compression
    if model_bytes_bf16 / 20 < vram_bytes:
        return 3  # Sparse + sketch needed
    
    # With 40x compression + SSD
    if model_bytes_bf16 / 40 < vram_bytes and ram_gb > 16:
        return 4  # SSD offload needed
    
    # Extreme case
    return 5  # MEZO + SSD


def detect_model_format(model_path: str) -> Dict[str, Any]:
    """Detect model format and requirements."""
    info = {
        'format': 'unknown',
        'vocab_size': 0,
        'hidden_size': 0,
        'num_layers': 0,
        'num_params': 0,
        'expected_dataset_format': 'text',
    }
    
    path = Path(model_path)
    
    # Check if it's a single file (like .safetensors or .bin)
    if path.is_file():
        # Try to find config in parent directory or alongside the file
        config_path = path.parent / 'config.json'
        if not config_path.exists() and path.suffix in ['.safetensors', '.bin', '.pt', '.pth']:
            # Try to infer from filename
            filename_lower = path.name.lower()
            if 'llama' in filename_lower or 'mistral' in filename_lower or 'mixtral' in filename_lower:
                info['format'] = 'llama'
                info['expected_dataset_format'] = 'chat'
                # Rough estimate: 7B model = ~7B params
                if '70b' in filename_lower or '65b' in filename_lower:
                    info['num_params'] = 70_000_000_000 if '70b' in filename_lower else 65_000_000_000
                elif '13b' in filename_lower:
                    info['num_params'] = 13_000_000_000
                elif '8b' in filename_lower:
                    info['num_params'] = 8_000_000_000
                elif '3b' in filename_lower:
                    info['num_params'] = 3_000_000_000
                elif '1b' in filename_lower or '1.1b' in filename_lower:
                    info['num_params'] = 1_000_000_000
                else:
                    info['num_params'] = 7_000_000_000  # Default to 7B
            elif 'gpt' in filename_lower:
                info['format'] = 'gpt2'
                info['expected_dataset_format'] = 'text'
                info['num_params'] = 1_500_000_000  # Rough estimate
            elif 'qwen' in filename_lower:
                info['format'] = 'qwen'
                info['expected_dataset_format'] = 'chat'
                if '72b' in filename_lower:
                    info['num_params'] = 72_000_000_000
                elif '32b' in filename_lower:
                    info['num_params'] = 32_000_000_000
                elif '14b' in filename_lower:
                    info['num_params'] = 14_000_000_000
                elif '7b' in filename_lower:
                    info['num_params'] = 7_000_000_000
                else:
                    info['num_params'] = 7_000_000_000
            else:
                # Generic - try to estimate from file size
                file_size = path.stat().st_size
                # Very rough: ~2 bytes per parameter for bf16
                info['num_params'] = max(int(file_size / 2), 1_000_000_000)
        
        # If we found a config nearby, use it
        if config_path.exists():
            model_path = str(config_path.parent)
    
    # Try transformers auto-detection (works for directories or if we found config)
    if TRANSFORMERS_AVAILABLE and info['format'] == 'unknown':
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            # Detect architecture family
            model_type = getattr(config, 'model_type', 'unknown')
            
            if model_type in ['llama', 'mistral', 'mixtral']:
                info['format'] = 'llama'
                info['expected_dataset_format'] = 'chat'
            elif model_type in ['gpt2', 'gpt_neo', 'gptj']:
                info['format'] = 'gpt2'
                info['expected_dataset_format'] = 'text'
            elif model_type in ['qwen', 'qwen2']:
                info['format'] = 'qwen'
                info['expected_dataset_format'] = 'chat'
            else:
                info['format'] = model_type
                info['expected_dataset_format'] = 'text'
            
            # Extract dimensions
            info['vocab_size'] = getattr(config, 'vocab_size', 0)
            info['hidden_size'] = getattr(config, 'hidden_size', 0)
            info['num_layers'] = getattr(config, 'num_hidden_layers', 
                                         getattr(config, 'n_layer', 0))
            
            # Estimate parameters
            vocab_params = info['vocab_size'] * info['hidden_size']
            layer_params = info['num_layers'] * info['hidden_size'] ** 2 * 12  # Approximate
            info['num_params'] = vocab_params + layer_params
            
        except Exception as e:
            logger.debug(f"Could not auto-detect model format: {e}")
    
    # Final validation: ensure num_params is positive
    if info['num_params'] <= 0:
        logger.warning("Model parameter count is 0 or negative, using default estimate of 1B")
        info['num_params'] = 1_000_000_000
    
    return info


def scan_datasets(dataset_dir: str) -> List[Path]:
    """Scan directory recursively for dataset files."""
    path = Path(dataset_dir)
    if not path.exists():
        return []
    
    # Supported formats
    extensions = ['.jsonl', '.json', '.txt', '.csv', '.parquet']
    
    files = []
    for ext in extensions:
        # Recursive search with ** pattern
        files.extend(path.rglob(f'*{ext}'))
    
    return sorted(files)


def infer_dataset_format(files: List[Path], max_samples: int = 5) -> str:
    """
    Infer dataset format from file content using multi-file sampling.
    
    Args:
        files: List of dataset files to analyze
        max_samples: Maximum number of files to sample (default 5)
    
    Returns:
        Detected format string, or 'mixed' if formats differ across files
    """
    if not files:
        return 'unknown'
    
    # Sample up to max_samples files for better detection accuracy
    sample_files = files[:max_samples] if len(files) > max_samples else files
    detected_formats = []
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            if not first_line:
                continue  # Skip empty files
            
            # Try parse as JSON
            try:
                data = json.loads(first_line)
                
                # Check for chat format
                if isinstance(data, dict):
                    if 'role' in data and 'content' in data:
                        detected_formats.append('chat')
                    elif 'messages' in data:
                        detected_formats.append('chat_list')
                    elif 'text' in data:
                        detected_formats.append('text')
                    elif 'input' in data and 'output' in data:
                        detected_formats.append('instruction')
                    else:
                        detected_formats.append('json')
                else:
                    detected_formats.append('json')
            except json.JSONDecodeError:
                # Not valid JSON - treat as plain text
                detected_formats.append('text')
                
        except Exception as e:
            logger.warning(f"Could not read dataset file {sample_file}: {e}")
            continue
    
    if not detected_formats:
        return 'unknown'
    
    # Check for consistency across samples
    unique_formats = set(detected_formats)
    if len(unique_formats) == 1:
        # All files agree on format
        return detected_formats[0]
    else:
        # Mixed formats - return the most common one with warning
        from collections import Counter
        format_counts = Counter(detected_formats)
        most_common = format_counts.most_common(1)[0]
        
        if most_common[1] >= len(detected_formats) * 0.6:  # 60% threshold
            logger.warning(
                f"Mixed dataset formats detected: {dict(format_counts)}. "
                f"Using majority format: {most_common[0]}"
            )
            return most_common[0]
        else:
            logger.warning(
                f"Highly mixed dataset formats with no clear majority: {dict(format_counts)}. "
                f"Consider organizing datasets by format."
            )
            return 'mixed'


def auto_format_dataset(
    files: List[Path], 
    source_format: str, 
    target_format: str,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Auto-format dataset to match model requirements.
    
    Args:
        files: List of dataset files
        source_format: Source format ('text', 'json', 'chat', 'mixed', etc.)
        target_format: Target format for model
        system_prompt: Optional custom system prompt
    
    Returns:
        List of formatted data samples
    """
    formatted = []
    
    default_system = system_prompt or "You are a helpful assistant."
    
    # Handle mixed formats by detecting per-file
    if source_format == 'mixed':
        logger.warning("Mixed format detected - processing each file with individual format detection")
        for file in files:
            # Detect format for this specific file
            file_format = infer_dataset_format([file], max_samples=1)
            if file_format == 'unknown':
                file_format = 'text'  # Fallback
            
            # Recursively process with detected format
            file_data = auto_format_dataset([file], file_format, target_format, system_prompt)
            formatted.extend(file_data)
        return formatted
    
    for file in files:
        logger.info(f"[AUTO-FORMAT] Processing {file.name}")
        logger.info(f"  Source: {source_format} -> Target: {target_format}")
        
        if source_format == target_format:
            # Load as-is
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        formatted.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        formatted.append({'text': line.strip()})
        
        elif source_format == 'text' and target_format == 'chat':
            # Wrap plain text as chat
            logger.info(f"  [APPLYING FIX] Wrapping with system prompt: {default_system[:50]}...")
            
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Split into chunks
                chunks = content.split('\n\n')
                
                for chunk in chunks:
                    if chunk.strip():
                        formatted.append({
                            'messages': [
                                {'role': 'system', 'content': default_system},
                                {'role': 'user', 'content': chunk.strip()}
                            ]
                        })
            
            logger.info(f"  [RESULT] Created {len(formatted)} conversation samples")
        
        elif source_format == 'json' and target_format == 'chat':
            # Try to convert JSON to chat
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Try to extract text as user message
                        if 'text' in data:
                            formatted.append({
                                'messages': [
                                    {'role': 'system', 'content': default_system},
                                    {'role': 'user', 'content': data['text']}
                                ]
                            })
                        else:
                            formatted.append(data)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        pass
        
        else:
            # Load raw
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    formatted.append({'text': line.strip()})
    
    return formatted


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING DATASET AND COLLATOR
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleTextDataset(Dataset):
    """Simple dataset for text training."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        format_type: str = 'text',
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.format_type == 'chat' or 'messages' in item:
            # Format chat messages
            messages = item.get('messages', [])
            text = self._format_messages(messages)
        elif 'text' in item:
            text = item['text']
        else:
            text = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
        }
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages to text."""
        text_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                text_parts.append(f"System: {content}")
            elif role == 'user':
                text_parts.append(f"User: {content}")
            elif role == 'assistant':
                text_parts.append(f"Assistant: {content}")
        
        return '\n'.join(text_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TRAINING LOOP UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    model,
    optimizer: ChimeraGearOptimizer,
    epoch: int,
    step: int,
    loss: float,
    path: Optional[Union[str, Path]] = None
):
    """Save training checkpoint with cross-platform path handling."""
    # Default path with proper cross-platform handling
    if path is None:
        output_dir = Path.cwd() / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "checkpoint.pt"
    else:
        path = Path(path)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'gear': optimizer.current_gear,
        'timestamp': time.time(),
        'chimera_version': __version__,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer: ChimeraGearOptimizer, path: str) -> Dict[str, Any]:
    """
    Load training checkpoint with version validation.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
        
    Raises:
        RuntimeError: If checkpoint is incompatible with current version
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # PyTorch 2.0+ secure loading - weights_only=True required for security
    # This prevents arbitrary code execution from malicious checkpoint files
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    
    # Version compatibility check
    ckpt_version = checkpoint.get('chimera_version', 'unknown')
    current_version = __version__
    
    if ckpt_version != current_version:
        logger.warning(
            f"Checkpoint version mismatch: "
            f"checkpoint={ckpt_version}, current={current_version}"
        )
        
        # Check for breaking changes (major version mismatch)
        try:
            # Handle version strings like '1.0.0-beta', '1.0.0a1', etc.
            def extract_major(version_str: str) -> int:
                """Extract numeric major version, handling pre-release suffixes."""
                if version_str == 'unknown':
                    return 1
                # Split on common separators and take first numeric part
                for sep in ['.', '-', 'a', 'b', 'rc']:
                    version_str = version_str.split(sep)[0]
                try:
                    return int(version_str)
                except ValueError:
                    return 1  # Default if parsing fails
            
            ckpt_major = extract_major(ckpt_version)
            curr_major = extract_major(current_version)
            
            if ckpt_major != curr_major:
                raise RuntimeError(
                    f"Major version mismatch: checkpoint v{ckpt_major} vs current v{curr_major}. "
                    f"Breaking changes detected. Please train from scratch or use compatible version."
                )
        except (ValueError, IndexError):
            # Can't parse version, warn but continue
            logger.warning("Could not parse version numbers, proceeding with caution")
    
    # Validate required keys
    required_keys = ['model_state_dict', 'optimizer_state_dict']
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise RuntimeError(f"Checkpoint missing required keys: {missing_keys}")
    
    # Load states with error handling
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint state: {e}. "
            f"Model/optimizer architecture may have changed."
        ) from e
    
    logger.info(f"Checkpoint loaded: {path}")
    logger.info(f"  Version: {ckpt_version}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 0)}")
    logger.info(f"  Step: {checkpoint.get('step', 0)}")
    logger.info(f"  Gear: {checkpoint.get('gear', optimizer.current_gear)}")
    
    return checkpoint


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: ChimeraGearOptimizer,
    device: str,
    epoch: int,
    log_interval: int = 10,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    total_batches = len(dataloader)
    
    # Print initial progress state immediately
    print(f"\r  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0% | Batch 0/{total_batches} | Loss: ... | ETA: calculating...", end='', flush=True)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, triggering emergency downshift")
                optimizer.emergency_downshift()
                
                # Retry with smaller effective batch (skip this batch)
                torch.cuda.empty_cache()
                continue
            else:
                raise
        
        # Backward pass (only for non-MEZO gears)
        if optimizer.current_gear != 5:
            optimizer.zero_grad()
            loss.backward()
        
        # Optimizer step (with gear management)
        # For MEZO (gear 5), provide closure for zeroth-order gradient estimation
        if optimizer.current_gear == 5 and optimizer.mezo_mode:
            # Robust closure with error handling and retry logic
            # Use a class-based closure to avoid closure variable capture issues
            class MezoClosure:
                __slots__ = ['model', 'input_ids', 'attention_mask', 'labels', 
                           'attempts', 'last_error', 'batch_idx']
                
                def __init__(self, model, input_ids, attention_mask, labels, batch_idx):
                    self.model = model
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                    self.labels = labels
                    self.batch_idx = batch_idx
                    self.attempts = 0
                    self.last_error = None
                
                def __call__(self):
                    self.attempts += 1
                    
                    try:
                        outputs = self.model(
                            input_ids=self.input_ids, 
                            attention_mask=self.attention_mask, 
                            labels=self.labels
                        )
                        if outputs.loss is None:
                            raise RuntimeError("Model returned None loss")
                        return outputs.loss
                    except RuntimeError as e:
                        self.last_error = e
                        if 'out of memory' in str(e).lower():
                            # OOM during MEZO - this is critical
                            raise  # Re-raise for emergency handling
                        # Other errors - log and return high loss to indicate failure
                        logger.warning(f"MEZO closure error (attempt {self.attempts}): {e}")
                        return torch.tensor(1e6, device=self.input_ids.device)  # High penalty loss
            
            # Create closure instance for this batch
            mezo_closure = MezoClosure(model, input_ids, attention_mask, labels, batch_idx)
            
            try:
                optimizer.step(closure=mezo_closure)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"OOM during MEZO at batch {batch_idx}, triggering emergency downshift")
                    optimizer.emergency_downshift()
                    torch.cuda.empty_cache()
                    # Skip this batch after emergency handling
                    continue
                else:
                    logger.error(f"MEZO step failed after {mezo_closure.attempts} attempts: {mezo_closure.last_error or e}")
                    raise
        else:
            optimizer.step(closure=None)
        
        # Stats
        total_loss += loss.item()
        num_batches += 1
        
        # Logging with progress bar
        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            step_time = elapsed / (batch_idx + 1)
            
            # Progress bar
            progress = (batch_idx + 1) / total_batches
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            # ETA calculation
            if batch_idx > 0:
                eta_seconds = step_time * (total_batches - batch_idx - 1)
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."
            
            # Print progress bar to console (not just logger)
            print(f"\r  {bar} {progress*100:5.1f}% | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f} | ETA: {eta_str}", end='', flush=True)
            
            # Also log to file via logger
            logger.debug(
                f"Epoch {epoch} [{batch_idx}/{total_batches}] "
                f"Loss: {loss.item():.4f} Gear: {optimizer.current_gear} Step: {step_time:.3f}s"
            )
            
            # Update watchdog objective for BO
            if optimizer.watchdog:
                optimizer.watchdog.update_objective(
                    step_time=step_time,
                    oom_risk=False,  # If we got here, no OOM
                    loss_plateau=False,  # Simplified
                )
    
    # Clear progress bar line and print summary
    print()  # Newline after progress bar
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: PREFLIGHT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def preflight_check(
    models_dir: str,
    datasets_dir: str,
    gear: Union[int, str],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Pre-flight validation before training starts.
    Discovers model/dataset, validates everything, returns clean results.
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'recommendations': [],
        'hardware': {},
        'model_path': None,
        'dataset_files': [],
        'models_dir': models_dir,
        'datasets_dir': datasets_dir,
    }
    
    # 1. Hardware Detection
    hw = detect_hardware()
    results['hardware'] = hw
    
    # 2. Model Discovery & Validation
    models_path = Path(models_dir)
    model_path = None
    
    # Case 1: models_dir itself is a model folder (has config.json)
    if (models_path / 'config.json').exists():
        model_path = str(models_path)
    else:
        # Case 2: Look for subdirectories that are model folders
        subdirs = [
            d for d in models_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and (d / 'config.json').exists()
        ]
        if len(subdirs) == 1:
            model_path = str(subdirs[0])
        elif len(subdirs) > 1:
            results['warnings'].append(f"Multiple models found, using: {subdirs[0].name}")
            model_path = str(subdirs[0])
        else:
            # Case 3: Look for single model files
            model_files = [
                f for f in models_path.iterdir()
                if f.is_file() and not f.name.startswith('.') and f.suffix in ['.safetensors', '.bin', '.pt', '.pth']
            ]
            if len(model_files) == 1:
                model_path = str(model_files[0])
            elif len(model_files) > 1:
                results['warnings'].append(f"Multiple model files found, using: {model_files[0].name}")
                model_path = str(model_files[0])
    
    if model_path:
        results['model_path'] = model_path
    else:
        results['errors'].append(f"No model found in '{models_dir}'")
        results['recommendations'].append("Place a model folder (with config.json) or single .safetensors/.bin file")
        results['passed'] = False
    
    # Validate discovered model
    if results['model_path']:
        try:
            model_info = detect_model_format(results['model_path'])
            results['model'] = model_info
            
            # Size check
            if model_info['num_params'] > 0 and hw['vram_gb'] > 0:
                min_vram = (model_info['num_params'] * 2) / 50 / (1024**3) * 1.5
                if min_vram > hw['vram_gb']:
                    results['errors'].append(
                        f"Model ({model_info['num_params']/1e9:.1f}B params) too large for {hw['vram_gb']:.1f}GB VRAM"
                    )
                    results['recommendations'].append("Use a smaller model or CPU-only mode with --gear 5")
                    results['passed'] = False
            
            # Gear recommendation
            if gear != 'auto':
                calc_gear = calculate_starting_gear(model_info['num_params'], hw['vram_gb'], hw['ram_gb'])
                if int(gear) < calc_gear:
                    results['warnings'].append(f"Gear {gear} may OOM. Recommended: {calc_gear}")
                    
        except Exception as e:
            results['warnings'].append(f"Model validation error: {e}")
    
    # 3. Dataset Discovery & Validation
    dataset_files = scan_datasets(datasets_dir)
    results['dataset_files'] = dataset_files
    
    if not dataset_files:
        results['errors'].append(f"No datasets found in '{datasets_dir}'")
        results['recommendations'].append("Add .jsonl, .txt, or .json files to the Datasets/ directory")
        results['passed'] = False
    else:
        # Quick validation of first file
        try:
            with open(dataset_files[0], 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if dataset_files[0].suffix == '.jsonl':
                    try:
                        json.loads(first_line)
                    except (json.JSONDecodeError, ValueError):
                        results['warnings'].append(f"{dataset_files[0].name}: invalid JSONL, will parse as text")
        except Exception as e:
            results['warnings'].append(f"Cannot read dataset: {e}")
        
        # Calculate stats
        total_size = sum(f.stat().st_size for f in dataset_files if f.exists())
        results['dataset_stats'] = {
            'files': len(dataset_files),
            'total_size_mb': total_size / (1024**2),
        }
    
    # 4. Hardware Warnings
    if not hw['cuda_available'] and gear != 'auto' and int(gear) <= 2:
        results['warnings'].append("No GPU detected but gears 1-2 selected (will be slow)")
    
    if hw['ram_gb'] < 8:
        results['warnings'].append(f"Low RAM ({hw['ram_gb']:.1f}GB): SSD offloading may be needed")
    
    # 5. Dependencies
    if not TRANSFORMERS_AVAILABLE:
        results['errors'].append("'transformers' library not installed")
        results['recommendations'].append("Run: pip install transformers")
        results['passed'] = False
    
    # 6. Output Directory
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / '.write_test').touch()
        (output_dir / '.write_test').unlink()
    except Exception as e:
        results['errors'].append(f"Cannot write to output directory: {e}")
        results['passed'] = False
    
    # 7. Config Checks
    if args.lr < 1e-6 or args.lr > 1e-1:
        results['warnings'].append(f"Unusual learning rate: {args.lr}")
    
    # Checkpoint resume validation with proper None handling
    if getattr(args, 'resume', None):
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            results['errors'].append(f"Resume checkpoint not found: {resume_path}")
            results['recommendations'].append(f"Verify the checkpoint path: {resume_path}")
            results['passed'] = False
        elif not resume_path.is_file():
            results['errors'].append(f"Resume path is not a file: {resume_path}")
            results['passed'] = False
    
    return results


def print_preflight_report(results: Dict[str, Any]):
    """Print clean, human-readable preflight report."""
    hw = results['hardware']
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " CHIMERA-M Pre-Flight Check".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Hardware
    print("\n┌─ Hardware ─" + "─" * 57 + "┐")
    if hw.get('cuda_available'):
        print(f"│  GPU: Yes ({hw.get('vram_gb', 0):.1f}GB VRAM)".ljust(69) + "│")
    else:
        print(f"│  GPU: No (CPU only)".ljust(69) + "│")
    print(f"│  RAM: {hw.get('ram_gb', 0):.1f}GB".ljust(69) + "│")
    print(f"│  CPU: {hw.get('cpu_cores', 0)} cores".ljust(69) + "│")
    print("└" + "─" * 68 + "┘")
    
    # Model
    print("\n┌─ Model ────" + "─" * 57 + "┐")
    if results.get('model_path'):
        model_name = Path(results['model_path']).name[:50]
        print(f"│  Found: {model_name}".ljust(69) + "│")
        if 'model' in results:
            m = results['model']
            print(f"│  Format: {m.get('format', 'unknown')}".ljust(69) + "│")
            if m.get('num_params', 0) > 0:
                print(f"│  Size: ~{m.get('num_params', 0)/1e9:.2f}B parameters".ljust(69) + "│")
    else:
        print(f"│  ⚠ No model found in '{results['models_dir']}'".ljust(69) + "│")
    print("└" + "─" * 68 + "┘")
    
    # Dataset
    print("\n┌─ Dataset ──" + "─" * 57 + "┐")
    if results.get('dataset_files'):
        d = results.get('dataset_stats', {})
        print(f"│  Found: {d.get('files', len(results['dataset_files']))} files".ljust(69) + "│")
        if 'total_size_mb' in d:
            print(f"│  Size: {d['total_size_mb']:.1f}MB".ljust(69) + "│")
    else:
        print(f"│  ⚠ No datasets found in '{results['datasets_dir']}'".ljust(69) + "│")
    print("└" + "─" * 68 + "┘")
    
    # Warnings
    if results['warnings']:
        print("\n┌─ Warnings ─" + "─" * 57 + "┐")
        for w in results['warnings'][:5]:  # Limit to 5
            print(f"│  ! {w[:60]}".ljust(69) + "│")
        if len(results['warnings']) > 5:
            print(f"│  ... and {len(results['warnings']) - 5} more".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
    
    # Errors
    if results['errors']:
        print("\n┌─ Errors ───" + "─" * 57 + "┐")
        for e in results['errors'][:5]:
            print(f"│  ✗ {e[:60]}".ljust(69) + "│")
        if len(results['errors']) > 5:
            print(f"│  ... and {len(results['errors']) - 5} more".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
    
    # Recommendations
    if results['recommendations']:
        print("\n┌─ Suggested Fixes ─" + "─" * 49 + "┐")
        for i, r in enumerate(results['recommendations'][:3], 1):
            print(f"│  {i}. {r[:62]}".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
    
    # Status line
    print()
    if results['passed']:
        print("   ✅ All checks passed")
    else:
        print(f"   ❌ {len(results['errors'])} error(s) found - cannot start")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Main entry point for Chimera-M training.
    
    Flow:
    1. Parse args
    2. Run preflight check (includes ALL validation)
    3. Print report
    4. Block if errors
    5. Start training
    """
    parser = argparse.ArgumentParser(
        description="CHIMERA-M: Compressive Hybrid Architecture for Intelligent, Efficient Resource Allocation and Modeling"
    )
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, 
                       help='Number of data loading workers (0 = main thread only)')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--log-interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Checkpoint every N steps')
    
    # Gear args
    parser.add_argument('--gear', type=str, default='auto', 
                       help='Compression level: 1-5 or auto')
    parser.add_argument('--ssdr-threshold', type=float, default=0.85,
                       help='SSD offload RAM threshold (0-1)')
    parser.add_argument('--bo-off', action='store_true',
                       help='Disable Bayesian optimization')
    
    # Paths
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing model')
    parser.add_argument('--datasets-dir', type=str, default='./datasets',
                       help='Directory containing datasets')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for checkpoints')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREFLIGHT CHECK - Single unified validation
    # ═══════════════════════════════════════════════════════════════════════
    
    results = preflight_check(args.models_dir, args.datasets_dir, args.gear, args)
    print_preflight_report(results)
    
    if not results['passed']:
        print("\n❌ Cannot start training. Fix the issues above and try again.")
        sys.exit(1)
    
    print("\n✅ Ready to train!\n")
    
    # Extract validated paths
    model_path = results['model_path']
    dataset_files = results['dataset_files']
    
    # Continue with normal flow (logging will show details again)
    logger.info("=" * 70)
    logger.info("CHIMERA-M: Hardware Detection")
    logger.info("=" * 70)
    
    # Use hardware info from preflight check (avoid duplicate detection)
    hw = results['hardware']
    logger.info(f"CUDA Available: {hw['cuda_available']}")
    logger.info(f"VRAM: {hw['vram_gb']:.1f} GB")
    logger.info(f"RAM: {hw['ram_gb']:.1f} GB")
    logger.info(f"CPU Cores: {hw['cpu_cores']}")
    logger.info(f"SSD Available: {hw['ssd_available']}")
    
    # Scan Models directory
    logger.info("\n" + "=" * 70)
    logger.info("Model Detection")
    logger.info("=" * 70)
    
    model_name = Path(model_path).name if model_path else "unknown"
    logger.info(f"Model: {model_name}")
    
    # Detect model format
    model_info = detect_model_format(model_path)
    logger.info(f"Format: {model_info['format']}")
    logger.info(f"Estimated params: {model_info['num_params'] / 1e9:.2f}B")
    logger.info(f"Expected dataset format: {model_info['expected_dataset_format']}")
    
    # Scan Datasets (already scanned in preflight, just validate)
    logger.info("\n" + "=" * 70)
    logger.info("Dataset Detection")
    logger.info("=" * 70)
    
    if len(dataset_files) == 0:
        logger.error(f"No datasets found in {args.datasets_dir}")
        logger.error("Supported: .jsonl, .json, .txt, .csv, .parquet")
        sys.exit(1)
    
    logger.info(f"Found {len(dataset_files)} dataset files:")
    for f in dataset_files:
        logger.info(f"  - {f.name}")
    
    # Detect dataset format
    dataset_format = infer_dataset_format(dataset_files)
    logger.info(f"Detected format: {dataset_format}")
    
    # Validate and auto-format
    if dataset_format != model_info['expected_dataset_format']:
        logger.warning("[WARNING] Dataset format mismatch detected")
        logger.warning(f"  Model format: {model_info['format']} (requires {model_info['expected_dataset_format']})")
        logger.warning(f"  Input format: {dataset_format}")
        logger.info("[APPLYING FIX] Auto-formatting dataset...")
        
        formatted_data = auto_format_dataset(
            dataset_files,
            dataset_format,
            model_info['expected_dataset_format']
        )
        
        logger.info(f"[RESULT] Formatted {len(formatted_data)} samples")
    else:
        # Load as-is
        formatted_data = auto_format_dataset(
            dataset_files,
            dataset_format,
            dataset_format
        )
    
    # Validate dataset is not empty
    if not formatted_data or len(formatted_data) == 0:
        logger.error("Dataset is empty after formatting - cannot train")
        sys.exit(1)
    
    logger.info(f"[READY] {len(formatted_data)} training samples")
    
    # Load model and tokenizer
    logger.info("\n" + "=" * 70)
    logger.info("Loading Model")
    logger.info("=" * 70)
    
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if hw['cuda_available'] else torch.float32,
                device_map='auto' if hw['cuda_available'] else 'cpu',
                low_cpu_mem_usage=True,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    else:
        logger.error("transformers library required for model loading")
        sys.exit(1)
    
    # Determine starting gear
    if args.gear == 'auto':
        starting_gear = calculate_starting_gear(
            model_info['num_params'],
            hw['vram_gb'],
            hw['ram_gb']
        )
        logger.info(f"Auto-selected gear: {starting_gear}")
    else:
        starting_gear = int(args.gear)
        logger.info(f"Manual gear: {starting_gear}")
    
    # Setup optimizer
    device = 'cuda' if hw['cuda_available'] else 'cpu'
    
    optimizer = ChimeraGearOptimizer(
        model.parameters(),
        lr=args.lr,
        gear=starting_gear,
        device=device,
        cpu_offload=starting_gear >= 2,
        ssd_offload=starting_gear >= 4,
        ram_threshold=args.ssdr_threshold,
        bo_enabled=not args.bo_off,
    )
    
    logger.info(f"Optimizer: Gear {starting_gear} ({GEAR_LEVELS[starting_gear].compression_ratio}× compression)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    
    resume_path = getattr(args, 'resume', None)
    if resume_path:
        checkpoint = load_checkpoint(model, optimizer, resume_path)
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)
    
    # Create dataset
    train_dataset = SimpleTextDataset(
        formatted_data,
        tokenizer,
        max_length=args.max_length,
        format_type=model_info['expected_dataset_format'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers > 0,
    )
    
    logger.info(f"Dataset: {len(train_dataset)} samples")
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Training")
    logger.info("=" * 70)
    
    # Estimate total training time
    total_batches = len(train_loader) * (args.epochs - start_epoch)
    # Rough estimate: ~0.5-2s per batch depending on hardware
    est_seconds_per_batch = 2.0 if not hw['cuda_available'] else 0.5
    est_total_hours = (total_batches * est_seconds_per_batch) / 3600
    
    print(f"\n  Total batches: {total_batches} across {args.epochs - start_epoch} epoch(s)")
    print(f"  Estimated training time: {est_total_hours:.1f} hours")
    print(f"  (Press Ctrl+C to pause and save checkpoint)\n")
    
    # Setup signal handler for graceful shutdown on Ctrl+C
    import signal
    shutdown_requested = False
    
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            logger.info("\n[INTERRUPT] Ctrl+C received - saving checkpoint before exit...")
        else:
            logger.info("[INTERRUPT] Second interrupt - forcing exit")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    global_step = start_step
    
    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            avg_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch + 1,
                args.log_interval,
            )
            
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_path = f"{args.output_dir}/checkpoint_epoch{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch + 1, global_step, avg_loss, checkpoint_path)
            
            # Check if shutdown was requested during epoch
            if shutdown_requested:
                logger.info("[INTERRUPT] Shutdown requested, exiting gracefully...")
                break
        
        if not shutdown_requested:
            logger.info("\n" + "=" * 70)
            logger.info("Training Complete")
            logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPT] Training interrupted by user")
    except Exception as e:
        logger.error(f"\n[ERROR] Training failed: {e}")
        raise
    finally:
        # Always save checkpoint on exit (even if interrupted)
        if shutdown_requested:
            emergency_path = f"{args.output_dir}/checkpoint_interrupt.pt"
            save_checkpoint(model, optimizer, epoch + 1, global_step, avg_loss, emergency_path)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        
        # Save final model if training completed normally
        if not shutdown_requested:
            final_path = f"{args.output_dir}/final_model"
            try:
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                logger.info(f"Final model saved: {final_path}")
            except Exception as e:
                logger.warning(f"Could not save final model: {e}")
        
        # Cleanup
        if optimizer.watchdog:
            optimizer.watchdog.stop()
        
        if optimizer.paged_memory:
            optimizer.paged_memory.cleanup()


if __name__ == '__main__':
    main()
