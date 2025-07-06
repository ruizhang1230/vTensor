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

#include <cuda.h>

#include <iostream>

#include <mutex>
#include <numeric>

#include <torch/torch.h>

#include "vtensor.h"

#include "cu_util.h"
#include "logging.h"


void init_shared_phy_blocks(int num_blocks, size_t block_size) {
  int device_id = -1;
  DRV_CALL(cuCtxGetDevice(&device_id));
  for (int i = 0; i < num_blocks; i++) {
    std::shared_ptr<PhyBlock> phy_block_pre =
        std::make_shared<PhyBlock>(device_id, block_size);
    if (phy_block_pre->status != CUDA_SUCCESS) {
      WARN(0, "init_shared_phy_blocks failed");
      return;
    }
    shared_phy_blocks_pre.emplace_back(phy_block_pre);
    std::shared_ptr<PhyBlock> phy_block_post =
        std::make_shared<PhyBlock>(device_id, block_size);
    if (phy_block_post->status != CUDA_SUCCESS) {
      WARN(0, "init_shared_phy_blocks failed");
      return;
    }
    shared_phy_blocks_post.emplace_back(phy_block_post);
  }
}

void init_unique_phy_blocks(int num_blocks, size_t block_size) {
  int device_id = -1;
  DRV_CALL(cuCtxGetDevice(&device_id));
  for (int i = 0; i < num_blocks; i++) {
    std::unique_ptr<PhyBlock> phy_block =
        std::make_unique<PhyBlock>(device_id, block_size);
    if (phy_block->status != CUDA_SUCCESS) {
      WARN(0, "init_unique_phy_blocks failed");
      return;
    }
    unique_phy_blocks.emplace_back(std::move(phy_block));
  }
}

void release_shared_phy_blocks() {
  int blocks_size = shared_phy_blocks_pre.size();
  for (int i = 0; i < blocks_size; i++) {
    auto tmp_pre = std::move(shared_phy_blocks_pre[blocks_size - i - 1]);
    shared_phy_blocks_pre.pop_back();
    auto tmp_post = std::move(shared_phy_blocks_post[blocks_size - i - 1]);
    shared_phy_blocks_post.pop_back();
  }
}

std::shared_ptr<Allocator> VmmTensor::allocator = nullptr;

VmmTensor::VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype,
                     int offset_index, int world_size, int pre_flag)
    : device_id(-1), used_size(0), world_size(world_size) {
  if (device_id == -1) {
    DRV_CALL(cuCtxGetDevice(&device_id));
  }

  size_t dtype_size = torch::elementSize(dtype);
  actual_size = std::accumulate(shape.begin(), shape.end(), dtype_size,
                                std::multiplies<int64_t>());

  if (this->allocator == nullptr) {
    this->allocator = nvgpu::VmmAllocator::instance();
  }

  this->allocator->reserve_virtual_addr((void **)&v_ptr, actual_size/*requested_size*/, &padded_size/*reserved_size*/, device_id, 0/*stream*/);

  std::cout << "[VmmTensor::VmmTensor] Reserving virtual address " << reinterpret_cast<uint64_t>((void *)v_ptr) << " with requested size " << actual_size << ", reserved_size " << padded_size << std::endl;

  AllocMemory(offset_index, world_size, pre_flag);

  tensor = GetTensor(shape, dtype);
}

VmmTensor::VmmTensor(std::vector<int64_t> shape, torch::Dtype dtype,
                     int offset_index, int world_size, Allocator* _allocator) : VmmTensor(shape, dtype, offset_index, world_size, 0/*pre_flag*/) {
   if (this->allocator == nullptr) {
      this->allocator = std::make_shared<nvgpu::VmmAllocator>();
   }
   this->allocator.reset(_allocator);
}

void VmmTensor::AllocMemory(int offset_index, int world_size, int pre_flag) {
  // Avoid concurrency issues caused by retries or others
  std::lock_guard<std::mutex> lock(mtx);

  size_t offset_size = padded_size / world_size;
  int shared_phy_index = 0;
  for (int i = 0; i < world_size; i++) {
    char *offset_addr = (char *)v_ptr + i * offset_size;
    if (i == offset_index) {
      if(unique_phy_blocks.size() >= 0) {
        this->u_p_block =
            std::move(unique_phy_blocks[unique_phy_blocks.size() - 1]);
        unique_phy_blocks.pop_back();
        this->allocator->exclusive_pool.add(this->u_p_block.get());
      } else {
        // use does not call init_shared_phy_blocks api, no pre allocated
        std::cout << "[AllocMemory] Not implmented yet" << std::endl;
        exit(0);
      }

      // use the address to find the block which own the address
      this->allocator->map_virtual_address(this->u_p_block.get(), (void *)offset_addr, offset_size);

      std::cout << "[AllocMemory]  map tensor::offset_index#" << offset_index << " with offset " << i * offset_size << " at " << reinterpret_cast<uint64_t>((void *)offset_addr) << " in block#" << this->u_p_block->block_id << std::endl;
    } else {
      // this tensor partition does not own the memory block
      std::shared_ptr<PhyBlock> phy_block;
      if (pre_flag) {
        assert(shared_phy_index < shared_phy_blocks_pre.size());
        phy_block = shared_phy_blocks_pre[shared_phy_index];
      } else {
        assert(shared_phy_index < shared_phy_blocks_post.size());
        phy_block = shared_phy_blocks_post[shared_phy_index];
      }

      this->allocator->shared_pool.add(phy_block.get());
      this->allocator->map_virtual_address(phy_block.get(), (void *)offset_addr, offset_size);

      std::cout << "[AllocMemory]  map shared tensor::offset_index#" << i << " with offset " << i * offset_size << " at " << reinterpret_cast<uint64_t>(offset_addr) << " in block#" << phy_block->block_id << std::endl;

      shared_phy_index++;
    }
  }
  used_size = actual_size;
}

torch::Tensor VmmTensor::SplitTensor(std::vector<int64_t> shape,
                                     torch::Dtype dtype, int offset_index) {
  if (offset_v_ptr != 0) {
    throw std::runtime_error("SplitTensor already called");
  }
  offset_size = padded_size / world_size;

  size_t reserved_size = 0;
  this->allocator->reserve_virtual_addr((void **)&offset_v_ptr, offset_size, &reserved_size, device_id, 0/*stream*/);

  std::cout << "[VmmTensor::SplitTensor] Reserving virtual address " << reinterpret_cast<uint64_t>((void *)offset_v_ptr) << " with requested size " << offset_size << ", reserved_size " << reserved_size << std::endl;

  this->allocator->map_virtual_address(this->u_p_block.get(), (void *)offset_v_ptr, offset_size);

  std::cout << "[VmmTensor::SplitTensor] map tensor::offset_index#" << offset_index << " with offset " << offset_size << " at " << reinterpret_cast<uint64_t>((void *)offset_v_ptr) << " in block#" << this->u_p_block->block_id << std::endl;

  std::vector<int64_t> stride(shape.size());
  stride[stride.size() - 1] = 1;
  for (int i = stride.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
  offset_tensor = torch::from_blob(
      reinterpret_cast<void *>(offset_v_ptr), shape, stride,
      [](void *offset_v_ptr) {}, options);

  return offset_tensor;
}

torch::Tensor VmmTensor::GetTensor() { return this->tensor; }

torch::Tensor VmmTensor::GetTensor(std::vector<int64_t> &shape,
                                   torch::Dtype dtype) {
  std::vector<int64_t> stride(shape.size());
  stride[stride.size() - 1] = 1;
  for (int i = stride.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
  torch::Tensor tensor = torch::from_blob(
      reinterpret_cast<void *>(v_ptr), shape, stride, [](void *v_ptr) {},
      options);

  return tensor;
}

VmmTensor::~VmmTensor() {
  if (v_ptr) {
    this->allocator->dealloc((void *)v_ptr, padded_size, device_id, 0/*stream*/);
  }

  if (offset_v_ptr != 0) {
    this->allocator->dealloc((void *)offset_v_ptr, offset_size, device_id, 0/*stream*/);
  }
  auto tmp = std::move(u_p_block);
}

// VMM torch tensor API

torch::Tensor vmm_realloc_tensor(void* address, std::vector<int64_t> shape, std::vector<int64_t> stride, torch::Dtype dtype, size_t request_size, int device, CUstream stream) {
  nvgpu::VmmAllocator::Ptr _allocator = nvgpu::VmmAllocator::instance();

  PhyBlock* block = _allocator->get_allocated_block(address);

  auto create_torch_tensor = [&](void* v_offset_addr) {
      torch::TensorOptions options =
          torch::TensorOptions().dtype(dtype).device(torch::kCUDA, block->device_id);

      torch::Tensor tensor = torch::from_blob(
          v_offset_addr, shape, stride, [](void *ptr) {}, options);

      return tensor;
  };

  std::shared_ptr<PhyBlock> _block = nullptr;
  auto find_available = [&](size_t size) {
      // find the nearest memory block
      PhyBlock* block = _allocator->owned_pool.find_available(size);

      if (block == nullptr) {
          _block = std::make_shared<PhyBlock>(device, size);
          block = _block.get();
      }

      return block;
  };

  if (block) {
    if (block->remaining_size > request_size) {
      size_t off = block->block_size - block->remaining_size;
      void* v_offset_addr = reinterpret_cast<void *>(reinterpret_cast<char *>(address) + off);

      _allocator->map_virtual_address(block, v_offset_addr, request_size);

      return create_torch_tensor(v_offset_addr);
    } else {
      CUdeviceptr d_ptr;

      // reserve virtual address
      size_t reserved_size = 0;
      _allocator->reserve_virtual_addr((void **)&d_ptr, request_size, &reserved_size, device, stream);

      // Note (yiakwy) : we reuse the remaining memroy in previous the most available block
      const size_t first_chunk_size = block->remaining_size;
      const size_t second_chunk_size = request_size - block->remaining_size;

      if (first_chunk_size > 0) {
        const size_t off = block->block_size - block->remaining_size;

        _allocator->map_virtual_address(block, (void *)d_ptr, first_chunk_size);
      }

      PhyBlock* one_available_block = find_available(second_chunk_size);

      void* v_offset_addr = reinterpret_cast<void *>(d_ptr + first_chunk_size);
      _allocator->map_virtual_address(one_available_block, v_offset_addr, second_chunk_size);

      if (_block != nullptr) {
        bool status = _allocator->owned_pool.add(_block);
        assert(status);
      }

      return create_torch_tensor((void *)d_ptr);
    }
  } else {
    CUdeviceptr d_ptr;

    // reserve virtual address
    size_t reserved_size = 0;
    _allocator->reserve_virtual_addr((void **)&d_ptr, request_size, &reserved_size, device, stream);

    PhyBlock* one_available_block = find_available(request_size);

    const size_t off = one_available_block->block_size - one_available_block->remaining_size;
    void* v_offset_addr = reinterpret_cast<void *>(d_ptr + off);

    _allocator->map_virtual_address(one_available_block, v_offset_addr, reserved_size);

    if (_block != nullptr) {
      bool status = _allocator->owned_pool.add(_block);
      assert(status);
    }

    return create_torch_tensor((void *)v_offset_addr);
  }
}
