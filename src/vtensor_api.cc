/* Copyright 2025 vTensor authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/extension.h>
#include <torch/torch.h>

#include "vtensor.h"
#include "allocator/vmm_allocator.h"

#ifdef __cplusplus
extern "C" { // Start C linkage block for C++ compilers
#endif

void* vmm_alloc(ssize_t size, int device, uintptr_t stream) {
  nvgpu::VmmAllocator::Ptr _allocator = nvgpu::VmmAllocator::instance();
  return _allocator->alloc((size_t)size, device, reinterpret_cast<CUstream>(stream));
}

void vmm_dealloc(int64_t address, size_t size, int device, uintptr_t stream) {
  nvgpu::VmmAllocator::Ptr _allocator = nvgpu::VmmAllocator::instance();
  return _allocator->dealloc(reinterpret_cast<void*>(address), size, device, reinterpret_cast<CUstream>(stream));
}

#ifdef __cplusplus
} // End C linkage block
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "vTensor";

  pybind11::class_<VmmTensor>(m, "tensor")
      .def(pybind11::init<std::vector<int64_t>, torch::Dtype, int, int, int>())
      .def("realloc_memory", &VmmTensor::AllocMemory)
      .def("split_tensor", &VmmTensor::SplitTensor)
      .def("to_torch_tensor", py::overload_cast<>(&VmmTensor::GetTensor));

  m.def("init_shared_phy_blocks", &init_shared_phy_blocks,
        "init_shared_phy_blocks");
  m.def("init_unique_phy_blocks", &init_unique_phy_blocks,
        "init_unique_phy_blocks");
  m.def("release_shared_phy_blocks", &release_shared_phy_blocks,
        "release_shared_phy_blocks");

  // VMM Allocator API

  pybind11::class_<nvgpu::VmmAllocator>(m, "vmm_allocator")
      .def(pybind11::init<>())
      .def("alloc", [](nvgpu::VmmAllocator& self, size_t size, int device, uintptr_t stream){
            return self.alloc(size, device, reinterpret_cast<CUstream>(stream));
      })
      .def("dealloc", [](nvgpu::VmmAllocator& self, int64_t address, size_t size, int device, uintptr_t stream){
            return self.dealloc(reinterpret_cast<void*>(address), size, device, reinterpret_cast<CUstream>(stream));
      });

  m.def("vmm_alloc", &vmm_alloc);
  m.def("vmm_dealloc", &vmm_dealloc);

  m.def("vmm_tensor", [](uintptr_t address, std::vector<int64_t> shape, std::vector<int64_t> stride, torch::Dtype dtype, int request_size, int device, uintptr_t stream) {
      return vmm_realloc_tensor(reinterpret_cast<void *>(address), shape, stride, dtype, request_size, device, reinterpret_cast<CUstream>(stream));
  });
}
