import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
except:
    raise Exception("Base env does not provide torch distribution. Exit.")

from helper_cuda import CUDA_HOME, _is_cuda, get_cuda_libraries

## Constant
SUPPORTED_DEVICES = [
    "CUDA",
]

PROJECT_ROOT = Path(__file__).parent.resolve()

operator_namespace = "vTensor"

include_dirs = [
    PROJECT_ROOT / "include" / "vtensor",
    PROJECT_ROOT / "csrc",
]

srcs = [
    "src/vtensor.cpp",
    "src/allocator/allocator.cpp",
    "src/allocator/expandable_phyblock.cpp",
    "src/allocator/vmm_allocator.cpp",
    "src/vtensor_api.cc",
]

cxx_flags = ["-O0", "-g"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]


def get_device_libs():
    if _is_cuda:
        return get_cuda_libraries()
    else:
        raise RuntimeError(
            f"Unknown runtime environment, only these {SUPPORTED_DEVICES} supported for the moment."
        )


setup(
    name="vTensor",
    version="0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name=f"{operator_namespace}.cpp_ext",
            sources=srcs,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
            },
            libraries=get_device_libs(),
            extra_link_args=extra_link_args,
            # py_limited_api="py39",
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    author="vTensor authors",
    author_email="@antgroup.com",
    description="VMM-based Tensor library for FlowMLA",
)
