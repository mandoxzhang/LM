import torch
import torch_musa

def synchronize():
    """Similar to cuda.synchronize().
    Waits for all kernels in all streams on a CUDA device to complete.
    """
    if torch_musa.is_available():
        torch_musa.synchronize()