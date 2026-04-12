"""
Setup for Ternary Codec C++ Extension
Build: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import platform

# Platform-specific flags
if platform.system() == 'Darwin':
    # macOS - Apple Clang (no -mtune=native support, no OpenMP)
    extra_compile = ["-O3", "-ffast-math"]
    extra_link = []
    
    # Only add x86-specific flags on Intel Macs
    import platform as platform_module
    machine = platform_module.machine()
    if machine == 'x86_64':
        extra_compile.extend(["-mavx", "-mavx2", "-mfma"])
else:
    # Linux with GCC
    extra_compile = ["-O3", "-fopenmp", "-march=native", "-ffast-math"]
    extra_link = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "ternary_codec",
        sources=["ternary_codec.cpp"],
        include_dirs=[pybind11.get_include()],
        cxx_std=14,
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
    ),
]

setup(
    name="ternary_codec",
    version="1.0.0",
    author="CHIMERA-M Team",
    description="Fast ternary weight quantization codec",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
