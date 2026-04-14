"""
C Extensions for CHIMERA-M
Fast implementations of performance-critical components

Components:
- count_min_sketch: Fast hash-based optimizer state compression
- dataset_fast: Fast JSONL parsing and preprocessing

Build:
    cd chimera_m_c
    python build.py
"""

import os
import sys
import ctypes
import warnings
import numpy as np
import torch
from pathlib import Path

# Try to load C extensions
_CMS_AVAILABLE = False
_DATASET_AVAILABLE = False

_cms_lib = None
_dataset_lib = None

def _load_cms_library():
    """Load Count-Min Sketch C library."""
    global _cms_lib, _CMS_AVAILABLE
    
    if _cms_lib is not None:
        return _CMS_AVAILABLE
    
    # Find library
    lib_dir = Path(__file__).parent
    lib_path = lib_dir / 'count_min_sketch.so'
    
    if not lib_path.exists():
        # Try alternate extensions
        for ext in ['.dylib', '.dll']:
            alt_path = lib_dir / f'count_min_sketch{ext}'
            if alt_path.exists():
                lib_path = alt_path
                break
    
    if not lib_path.exists():
        warnings.warn(
            f"Count-Min Sketch C library not found. "
            f"Build with: cd chimera_m_c && python build.py"
        )
        _CMS_AVAILABLE = False
        return False
    
    try:
        _cms_lib = ctypes.CDLL(str(lib_path))
        
        # Setup cms_update function
        # void cms_update(float* tables_m, float* tables_v, const int32_t* indices,
        #                 const float* values_m, const float* values_v, const int32_t* seeds,
        #                 int depth, int width, int n, float beta1, float beta2, int step_count)
        _cms_lib.cms_update.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # tables_m
            ctypes.POINTER(ctypes.c_float),  # tables_v
            ctypes.POINTER(ctypes.c_int32),  # indices
            ctypes.POINTER(ctypes.c_float),  # values_m
            ctypes.POINTER(ctypes.c_float),  # values_v
            ctypes.POINTER(ctypes.c_int32),  # seeds
            ctypes.c_int,  # depth
            ctypes.c_int,  # width
            ctypes.c_int,  # n
            ctypes.c_float,  # beta1
            ctypes.c_float,  # beta2
            ctypes.c_int,  # step_count
        ]
        _cms_lib.cms_update.restype = None
        
        # Setup cms_query function
        # void cms_query(const float* tables_m, const float* tables_v, const int32_t* indices,
        #                const int32_t* seeds, int depth, int width, int n,
        #                float* out_m, float* out_v)
        _cms_lib.cms_query.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # tables_m
            ctypes.POINTER(ctypes.c_float),  # tables_v
            ctypes.POINTER(ctypes.c_int32),  # indices
            ctypes.POINTER(ctypes.c_int32),  # seeds
            ctypes.c_int,  # depth
            ctypes.c_int,  # width
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # out_m
            ctypes.POINTER(ctypes.c_float),  # out_v
        ]
        _cms_lib.cms_query.restype = None
        
        # Setup cms_init_tables
        _cms_lib.cms_init_tables.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # tables_m
            ctypes.POINTER(ctypes.c_float),  # tables_v
            ctypes.c_int,  # depth
            ctypes.c_int,  # width
        ]
        _cms_lib.cms_init_tables.restype = None
        
        _CMS_AVAILABLE = True
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to load Count-Min Sketch C library: {e}")
        _CMS_AVAILABLE = False
        return False

def _load_dataset_library():
    """Load dataset preprocessing C library."""
    global _dataset_lib, _DATASET_AVAILABLE
    
    if _dataset_lib is not None:
        return _DATASET_AVAILABLE
    
    lib_dir = Path(__file__).parent
    lib_path = lib_dir / 'dataset_fast.so'
    
    if not lib_path.exists():
        for ext in ['.dylib', '.dll']:
            alt_path = lib_dir / f'dataset_fast{ext}'
            if alt_path.exists():
                lib_path = alt_path
                break
    
    if not lib_path.exists():
        _DATASET_AVAILABLE = False
        return False
    
    try:
        _dataset_lib = ctypes.CDLL(str(lib_path))
        
        # Setup parse_jsonl_batch function
        # int parse_jsonl_batch(const char* data, size_t len, char** outputs, size_t* output_lens, int max_lines)
        _dataset_lib.parse_jsonl_batch.argtypes = [
            ctypes.c_char_p,  # data
            ctypes.c_size_t,  # len
            ctypes.POINTER(ctypes.c_char_p),  # outputs array
            ctypes.POINTER(ctypes.c_size_t),  # output_lens
            ctypes.c_int,  # max_lines
        ]
        _dataset_lib.parse_jsonl_batch.restype = ctypes.c_int
        
        _DATASET_AVAILABLE = True
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to load dataset C library: {e}")
        _DATASET_AVAILABLE = False
        return False

# Try to load on import
_load_cms_library()
_load_dataset_library()

def cms_is_available():
    """Check if Count-Min Sketch C extension is available."""
    return _CMS_AVAILABLE

def dataset_is_available():
    """Check if dataset C extension is available."""
    return _DATASET_AVAILABLE

def _ensure_contiguous_cpu(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Ensure tensor is contiguous and on CPU for safe ctypes pointer access.
    
    Args:
        tensor: Input tensor
        dtype: Target dtype for conversion
    
    Returns:
        Contiguous CPU tensor with correct dtype
    
    Raises:
        RuntimeError: If tensor cannot be made contiguous/moved to CPU
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Move to CPU if needed
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # Convert dtype if needed
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    
    # Ensure contiguous memory layout
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    return tensor


def cms_update_fast(
    tables_m: torch.Tensor,
    tables_v: torch.Tensor,
    indices: torch.Tensor,
    values_m: torch.Tensor,
    values_v: torch.Tensor,
    seeds: torch.Tensor,
    beta1: float = 0.9,
    beta2: float = 0.999,
    step_count: int = 1
):
    """
    Fast Count-Min Sketch update using C extension.
    
    Args:
        tables_m: [depth, width] momentum table (updated in-place)
        tables_v: [depth, width] variance table (updated in-place)
        indices: [n] parameter indices
        values_m: [n] momentum values
        values_v: [n] variance values
        seeds: [depth] hash seeds
        beta1, beta2: Adam decay rates
        step_count: current step for bias correction (kept for API compatibility,
                   but bias correction is applied at query time, not update)
    
    Returns:
        True if C extension was used, False if PyTorch fallback needed
    """
    if not _CMS_AVAILABLE or _cms_lib is None:
        return False  # Signal to use Python fallback
    
    try:
        # Validate and prepare tensors for safe C pointer access
        tables_m_c = _ensure_contiguous_cpu(tables_m, torch.float32)
        tables_v_c = _ensure_contiguous_cpu(tables_v, torch.float32)
        indices_c = _ensure_contiguous_cpu(indices, torch.int32)
        values_m_c = _ensure_contiguous_cpu(values_m, torch.float32)
        values_v_c = _ensure_contiguous_cpu(values_v, torch.float32)
        seeds_c = _ensure_contiguous_cpu(seeds, torch.int32)
        
        # Validate shapes
        if tables_m_c.shape != tables_v_c.shape:
            raise ValueError(f"tables_m shape {tables_m_c.shape} != tables_v shape {tables_v_c.shape}")
        
        if tables_m_c.dim() != 2:
            raise ValueError(f"tables must be 2D, got {tables_m_c.dim()}D")
        
        depth, width = tables_m_c.shape
        n = indices_c.numel()
        
        if seeds_c.numel() != depth:
            raise ValueError(f"seeds length {seeds_c.numel()} != depth {depth}")
        
        if values_m_c.numel() != n or values_v_c.numel() != n:
            raise ValueError(f"values length must match indices length {n}")
        
        # Ensure all tensors are contiguous before getting pointers (Issue #40)
        tables_m_c = tables_m_c.contiguous()
        tables_v_c = tables_v_c.contiguous()
        indices_c = indices_c.contiguous()
        values_m_c = values_m_c.contiguous()
        values_v_c = values_v_c.contiguous()
        seeds_c = seeds_c.contiguous()
        
        # Validate tensors have valid memory before accessing
        if not (tables_m_c.is_contiguous() and tables_v_c.is_contiguous() and 
                indices_c.is_contiguous() and values_m_c.is_contiguous() and
                values_v_c.is_contiguous() and seeds_c.is_contiguous()):
            raise RuntimeError("Failed to make tensors contiguous for C extension")
        
        # Get C-compatible pointers
        tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        indices_ptr = ctypes.cast(indices_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        values_m_ptr = ctypes.cast(values_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        values_v_ptr = ctypes.cast(values_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        seeds_ptr = ctypes.cast(seeds_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        
        # Validate pointers are non-null
        if not all([tables_m_ptr, tables_v_ptr, indices_ptr, values_m_ptr, values_v_ptr, seeds_ptr]):
            raise RuntimeError("Failed to get valid memory pointers for C extension")
        
        # Call C function and check return value (Issue #39)
        result = _cms_lib.cms_update(
            tables_m_ptr, tables_v_ptr,
            indices_ptr, values_m_ptr, values_v_ptr, seeds_ptr,
            depth, width, n,
            beta1, beta2, step_count
        )
        
        # Check return value (non-zero indicates error)
        if result != 0:
            raise RuntimeError(f"C extension cms_update returned error code: {result}")
        
        # Sync back to original device if needed
        if tables_m.device.type != 'cpu':
            tables_m.copy_(tables_m_c)
        if tables_v.device.type != 'cpu':
            tables_v.copy_(tables_v_c)
        
        return True
        
    except Exception as e:
        warnings.warn(f"C extension update failed: {e}. Falling back to Python.")
        return False

def cms_query_fast(
    tables_m: torch.Tensor,
    tables_v: torch.Tensor,
    indices: torch.Tensor,
    seeds: torch.Tensor,
    beta1: float = 0.9,
    beta2: float = 0.999,
    step_count: int = 1
) -> tuple:
    """
    Fast Count-Min Sketch query using C extension.
    
    Args:
        tables_m: [depth, width] momentum table
        tables_v: [depth, width] variance table
        indices: [n] parameter indices to query
        seeds: [depth] hash seeds
        beta1, beta2: Adam decay rates for bias correction
        step_count: current step for bias correction (applied at query time)
    
    Returns:
        (m_hat, v_hat) tuple of tensors with bias correction applied, or None if fallback needed
    """
    if not _CMS_AVAILABLE or _cms_lib is None:
        return None  # Signal to use Python fallback
    
    try:
        # Validate and prepare tensors for safe C pointer access
        tables_m_c = _ensure_contiguous_cpu(tables_m, torch.float32)
        tables_v_c = _ensure_contiguous_cpu(tables_v, torch.float32)
        indices_c = _ensure_contiguous_cpu(indices, torch.int32)
        seeds_c = _ensure_contiguous_cpu(seeds, torch.int32)
        
        # Validate shapes
        if tables_m_c.shape != tables_v_c.shape:
            raise ValueError(f"tables_m shape {tables_m_c.shape} != tables_v shape {tables_v_c.shape}")
        
        if tables_m_c.dim() != 2:
            raise ValueError(f"tables must be 2D, got {tables_m_c.dim()}D")
        
        depth, width = tables_m_c.shape
        n = indices_c.numel()
        
        if seeds_c.numel() != depth:
            raise ValueError(f"seeds length {seeds_c.numel()} != depth {depth}")
        
        # Prepare output arrays on CPU
        out_m = torch.empty(n, dtype=torch.float32)
        out_v = torch.empty(n, dtype=torch.float32)
        
        # Get C-compatible pointers
        tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        indices_ptr = ctypes.cast(indices_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        seeds_ptr = ctypes.cast(seeds_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        out_m_ptr = ctypes.cast(out_m.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_v_ptr = ctypes.cast(out_v.data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        # Call C function to get min estimates
        _cms_lib.cms_query(
            tables_m_ptr, tables_v_ptr,
            indices_ptr, seeds_ptr,
            depth, width, n,
            out_m_ptr, out_v_ptr
        )
        
        # Apply bias correction at query time (matching Python CountMinSketch behavior)
        if step_count > 0:
            bias_correction1 = 1.0 - beta1 ** step_count
            bias_correction2 = 1.0 - beta2 ** step_count
            out_m = out_m / bias_correction1
            out_v = out_v / bias_correction2
        
        # Move to original device
        device = tables_m.device
        return out_m.to(device), out_v.to(device)
        
    except Exception as e:
        warnings.warn(f"C extension query failed: {e}. Falling back to Python.")
        return None

def cms_init_tables_fast(tables_m: torch.Tensor, tables_v: torch.Tensor):
    """Fast table initialization using C."""
    if not _CMS_AVAILABLE or _cms_lib is None:
        return False
    
    try:
        # Validate and prepare tensors
        tables_m_c = _ensure_contiguous_cpu(tables_m, torch.float32)
        tables_v_c = _ensure_contiguous_cpu(tables_v, torch.float32)
        
        if tables_m_c.shape != tables_v_c.shape:
            raise ValueError(f"Shape mismatch: {tables_m_c.shape} vs {tables_v_c.shape}")
        
        depth, width = tables_m_c.shape
        
        tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        _cms_lib.cms_init_tables(tables_m_ptr, tables_v_ptr, depth, width)
        
        # Sync back if needed
        if tables_m.device.type != 'cpu':
            tables_m.copy_(tables_m_c)
            tables_v.copy_(tables_v_c)
        
        return True
        
    except Exception as e:
        warnings.warn(f"C extension init failed: {e}. Using Python fallback.")
        return False

def parse_jsonl_fast(filepath: str, max_lines: int = 0) -> list:
    """
    Fast JSONL parsing using optimized Python.
    
    Args:
        filepath: path to .jsonl file
        max_lines: maximum lines to parse (0 = all)
    
    Returns:
        list of parsed dicts
    """
    import json
    
    results = []
    lines_read = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(data)
                lines_read += 1
                if max_lines > 0 and lines_read >= max_lines:
                    break
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
    
    return results

__all__ = [
    'cms_is_available',
    'dataset_is_available',
    'cms_update_fast',
    'cms_query_fast',
    'cms_init_tables_fast',
    'parse_jsonl_fast',
]
