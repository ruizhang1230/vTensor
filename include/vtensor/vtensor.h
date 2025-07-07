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

#pragma once

#include <cuda.h>
#include <iostream>
// #include <torch/extension.h>
// #include <torch/torch.h>
#include <vector>

#include "allocator/vmm_allocator.h"

using PhyBlock = nvgpu::ExpandablePhyBlock;
using Allocator = nvgpu::VmmAllocator;

torch::Tensor vmm_realloc_tensor(void * address, std::vector<int64_t> shape, std::vector<int64_t> stride, torch::Dtype dtype, size_t request_size, int device, CUstream stream);

void init_shared_phy_blocks(int num_blocks, size_t block_size);
void init_unique_phy_blocks(int num_blocks, size_t block_size);
void release_shared_phy_blocks();

class VmmTensor {
public:
  VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype, int offset_index,
            int world_size, int pre_flag);
  VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype, int offset_index,
            int world_size, Allocator* _allocator);
  ~VmmTensor();

  void AllocMemory(int offset_index, int world_size, int pre_flag);
  torch::Tensor GetTensor();
  torch::Tensor SplitTensor(std::vector<int64_t> shape, torch::Dtype dtype,
                            int offset_idnex);
  torch::Tensor GetTensor(std::vector<int64_t> &shape, torch::Dtype dtype);

private:
  int device_id;
  size_t padded_size;
  size_t actual_size;
  size_t used_size;
  int world_size;

  std::mutex mtx;

  torch::Tensor tensor;
  torch::Tensor offset_tensor;

  std::unique_ptr<PhyBlock> u_p_block;

  // can be shared cross different places (torch, this lib for example)
  static std::shared_ptr<Allocator> allocator;

  // NOTE : should this be owned by vTensor ?
  CUdeviceptr v_ptr;

  CUdeviceptr offset_v_ptr = 0;

  size_t offset_size = 0;
};

static std::vector<std::shared_ptr<PhyBlock>> shared_phy_blocks_pre;
static std::vector<std::shared_ptr<PhyBlock>> shared_phy_blocks_post;
static std::vector<std::unique_ptr<PhyBlock>> unique_phy_blocks;
