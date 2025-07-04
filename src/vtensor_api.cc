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
}
