"""Pre-load CUDA/cuDNN shared libs into the process before onnxruntime imports them.

LD_LIBRARY_PATH is read at process start by the dynamic linker, so changing it
from Python is too late. Instead we use ctypes.CDLL to dlopen the libraries
explicitly — subsequent dlopen calls by onnxruntime will find them already loaded.
"""

import ctypes
import os
from pathlib import Path

_SEARCH_DIRS = [
    "/usr/local/lib/ollama/mlx_cuda_v13",
    "/usr/local/lib/ollama/cuda_v13",
    "/usr/local/cuda/lib64",
    "/usr/lib64",
]

# Order matters: dependencies first
_LIBS = [
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn.so.9",
]


def preload() -> bool:
    loaded = []
    for lib in _LIBS:
        for d in _SEARCH_DIRS:
            path = os.path.join(d, lib)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(lib)
                except OSError:
                    pass
                break
    if loaded:
        print(f"[specter/cuda] preloaded: {', '.join(loaded)}")
        return True
    return False
