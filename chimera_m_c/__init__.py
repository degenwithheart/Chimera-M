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
        step_count: current step for bias correction
    
    Returns:
        True if C extension was used, False if PyTorch fallback
    """
    if not _CMS_AVAILABLE or _cms_lib is None:
        return False  # Signal to use Python fallback
    
    # Ensure contiguous CPU tensors
    tables_m_c = tables_m.contiguous().cpu()
    tables_v_c = tables_v.contiguous().cpu()
    indices_c = indices.contiguous().cpu().to(torch.int32)
    values_m_c = values_m.contiguous().cpu()
    values_v_c = values_v.contiguous().cpu()
    seeds_c = seeds.contiguous().cpu().to(torch.int32)
    
    depth, width = tables_m_c.shape
    n = indices_c.numel()
    
    # Get pointers
    tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    indices_ptr = ctypes.cast(indices_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    values_m_ptr = ctypes.cast(values_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    values_v_ptr = ctypes.cast(values_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    seeds_ptr = ctypes.cast(seeds_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    
    # Call C function
    _cms_lib.cms_update(
        tables_m_ptr, tables_v_ptr,
        indices_ptr, values_m_ptr, values_v_ptr, seeds_ptr,
        depth, width, n,
        beta1, beta2, step_count
    )
    
    # Copy back if needed (tables_m/v were modified in-place via C)
    if tables_m.device.type != 'cpu':
        tables_m.copy_(tables_m_c)
    if tables_v.device.type != 'cpu':
        tables_v.copy_(tables_v_c)
    
    return True

def cms_query_fast(
    tables_m: torch.Tensor,
    tables_v: torch.Tensor,
    indices: torch.Tensor,
    seeds: torch.Tensor
) -> tuple:
    """
    Fast Count-Min Sketch query using C extension.
    
    Args:
        tables_m: [depth, width] momentum table
        tables_v: [depth, width] variance table
        indices: [n] parameter indices to query
        seeds: [depth] hash seeds
    
    Returns:
        (m_hat, v_hat) tuple of tensors, or None if fallback needed
    """
    if not _CMS_AVAILABLE or _cms_lib is None:
        return None  # Signal to use Python fallback
    
    # Ensure contiguous CPU tensors
    tables_m_c = tables_m.contiguous().cpu()
    tables_v_c = tables_v.contiguous().cpu()
    indices_c = indices.contiguous().cpu().to(torch.int32)
    seeds_c = seeds.contiguous().cpu().to(torch.int32)
    
    depth, width = tables_m_c.shape
    n = indices_c.numel()
    
    # Prepare output arrays
    out_m = torch.empty(n, dtype=torch.float32)
    out_v = torch.empty(n, dtype=torch.float32)
    
    # Get pointers
    tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    indices_ptr = ctypes.cast(indices_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    seeds_ptr = ctypes.cast(seeds_c.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    out_m_ptr = ctypes.cast(out_m.data_ptr(), ctypes.POINTER(ctypes.c_float))
    out_v_ptr = ctypes.cast(out_v.data_ptr(), ctypes.POINTER(ctypes.c_float))
    
    # Call C function
    _cms_lib.cms_query(
        tables_m_ptr, tables_v_ptr,
        indices_ptr, seeds_ptr,
        depth, width, n,
        out_m_ptr, out_v_ptr
    )
    
    # Move to original device
    device = tables_m.device
    return out_m.to(device), out_v.to(device)

def cms_init_tables_fast(tables_m: torch.Tensor, tables_v: torch.Tensor):
    """Fast table initialization using C."""
    if not _CMS_AVAILABLE or _cms_lib is None:
        return False
    
    tables_m_c = tables_m.contiguous().cpu()
    tables_v_c = tables_v.contiguous().cpu()
    depth, width = tables_m_c.shape
    
    tables_m_ptr = ctypes.cast(tables_m_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    tables_v_ptr = ctypes.cast(tables_v_c.data_ptr(), ctypes.POINTER(ctypes.c_float))
    
    _cms_lib.cms_init_tables(tables_m_ptr, tables_v_ptr, depth, width)
    
    if tables_m.device.type != 'cpu':
        tables_m.copy_(tables_m_c)
        tables_v.copy_(tables_v_c)
    
    return True

def parse_jsonl_fast(filepath: str, max_lines: int = 0) -> list:
    """
    Fast JSONL parsing using C extension.
    
    Args:
        filepath: path to .jsonl file
        max_lines: maximum lines to parse (0 = all)
    
    Returns:
        list of parsed dicts, or None if C extension unavailable
    """
    if not _DATASET_AVAILABLE or _dataset_lib is None:
        return None
    
    # Read file
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # TODO: Implement full C integration
    # For now, fallback to Python
    return None

__all__ = [
    'cms_is_available',
    'dataset_is_available',
    'cms_update_fast',
    'cms_query_fast',
    'cms_init_tables_fast',
    'parse_jsonl_fast',
]
