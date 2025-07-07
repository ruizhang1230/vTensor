from typing import Callable

import torch
import vTensor.cpp_ext
from vTensor.cpp_ext import *


def __init_ctx__():
    pass


lib_file = vTensor.cpp_ext.__file__


def get_pluggable_allocator() -> torch.cuda.memory.CUDAPluggableAllocator:
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_file, "vmm_alloc", "vmm_dealloc"
    )
    return new_alloc


if __name__ == "__main__":
    pass
