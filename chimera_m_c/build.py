#!/usr/bin/env python3
"""
Build script for CHIMERA-M C extensions
Usage: python build.py
"""

import subprocess
import sys
import platform
from pathlib import Path

def find_compiler():
    """Find available C compiler."""
    compilers = ['gcc', 'clang', 'cc']
    for compiler in compilers:
        try:
            subprocess.run([compiler, '--version'], capture_output=True, check=True)
            return compiler
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None

def build_cms():
    """Build Count-Min Sketch C library with cross-platform support."""
    print("Building Count-Min Sketch extension...")
    
    src = Path("count_min_sketch.c")
    if not src.exists():
        print(f"Error: {src} not found")
        return False
    
    system = platform.system()
    compiler = find_compiler()
    
    if compiler is None:
        print("  ✗ No C compiler found (tried: gcc, clang, cc)")
        print("  Please install a C compiler and try again")
        return False
    
    print(f"  Using compiler: {compiler}")
    
    if system == 'Darwin':
        # macOS - build universal binary
        cmd = [
            compiler, '-O3', '-shared', '-fPIC',
            '-arch', 'x86_64', '-arch', 'arm64',  # Universal binary
            '-o', 'count_min_sketch.so',
            str(src),
            '-lm'
        ]
    elif system == 'Windows':
        # Windows - use .dll extension
        cmd = [
            compiler, '-O3', '-shared',
            '-o', 'count_min_sketch.dll',
            str(src)
        ]
    else:
        # Linux and other Unix-like systems
        cmd = [
            compiler, '-O3', '-shared', '-fPIC',
            '-o', 'count_min_sketch.so',
            str(src),
            '-lm'
        ]
        
        # Add optimizations only for GCC
        if compiler == 'gcc':
            cmd.insert(1, '-march=native')
            cmd.insert(2, '-ffast-math')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lib_name = 'count_min_sketch.dll' if system == 'Windows' else 'count_min_sketch.so'
            print(f"  ✓ {lib_name} built successfully")
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
