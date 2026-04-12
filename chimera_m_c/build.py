#!/usr/bin/env python3
"""
Build script for CHIMERA-M C extensions
Usage: python build.py
"""

import subprocess
import sys
import platform
from pathlib import Path

def build_cms():
    """Build Count-Min Sketch C library."""
    print("Building Count-Min Sketch extension...")
    
    src = Path("count_min_sketch.c")
    if not src.exists():
        print(f"Error: {src} not found")
        return False
    
    system = platform.system()
    
    if system == 'Darwin':
        # macOS - build universal binary
        cmd = [
            'gcc', '-O3', '-shared', '-fPIC',
            '-arch', 'x86_64', '-arch', 'arm64',  # Universal binary
            '-o', 'count_min_sketch.so',
            str(src),
            '-lm'
        ]
    else:
        # Linux
        cmd = [
            'gcc', '-O3', '-shared', '-fPIC',
            '-march=native',  # Optimize for current CPU
            '-ffast-math',
            '-o', 'count_min_sketch.so',
            str(src),
            '-lm'
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ count_min_sketch.so built successfully")
            return True
        else:
            print(f"  ✗ Build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Build error: {e}")
        return False

def build_dataset():
    """Build dataset preprocessing C library."""
    print("Building dataset preprocessing extension...")
    
    # For now, dataset uses pure Python fallback
    # Full C implementation would require jansson or similar JSON parser
    print("  ℹ Dataset C extension not yet implemented (using Python fallback)")
    return True

def main():
    """Build all C extensions."""
    print("=" * 60)
    print("Building CHIMERA-M C Extensions")
    print("=" * 60)
    print()
    
    success = True
    
    success = build_cms() and success
    success = build_dataset() and success
    
    print()
    if success:
        print("✓ All C extensions built successfully")
        print()
        print("To use:")
        print("  from chimera_m_c import cms_is_available, cms_update_fast")
        print("  if cms_is_available():")
        print("      cms_update_fast(tables_m, tables_v, ...)")
    else:
        print("✗ Some extensions failed to build")
        print("  Python fallbacks will be used instead")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
