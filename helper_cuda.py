import os
import re
import subprocess

import torch

try:
    from torch.utils.cpp_extension import CUDA_HOME
except:
    raise RuntimeError(
        "Base env does not provide Torch with support of CUDA SDK. Exit."
    )

CUDA_VERSION_PAT = r"CUDA version: (\S+)"
CUDA_SDK_ROOT = "/usr/local/cuda"
# VMM API is supported since CUDA 10.02 (ref to https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
ALLOWED_NVGPU_ARCHS = ["sm75", "sm_80", "sm_90", "sm_100", "sm_120"]


def is_cuda(cuda_sdk_root=None) -> bool:
    SDK_ROOT = f"{cuda_sdk_root or CUDA_SDK_ROOT}"

    def _check_sdk_installed() -> bool:
        # return True if this dir points to a directory or symbolic link
        return os.path.isdir(SDK_ROOT)

    if not _check_sdk_installed():
        return False

    # we provide torch for the base env, check whether it is valid installation
    result = subprocess.run(
        [
            f"/usr/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -o -m1 '.*'"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        print("Use CUDA pytorch, but no devices found!")
        return False, None

    sm_ver = result.stdout.strip()
    print(f"target NV gpu arch {sm_ver}")
    return True, sm_ver


_is_cuda = is_cuda()

if _is_cuda:
    assert CUDA_HOME is not None, "CUDA_HOME is not set"

    CUDA_HOME = os.environ.get("CUDA_HOME", CUDA_HOME)
    pass

cuda_libraries = [
    "cuda",
    # used by VMM API
    "cudart",
]


def get_cuda_libraries():
    return cuda_libraries
