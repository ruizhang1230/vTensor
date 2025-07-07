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

#include <atomic>

#include <cuda.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace nvgpu {

struct VmmAllocator;

template<typename BlockType>
struct BlockPool;

// Pytorch previously used Splitable Block which can be splitted to smaller addresses if the request size is smaller to the reserved size.
// This can reduce the number of the device allocation if we request a tensor with large size then split it to small sizes.
//
// With VMM API, we can request a tensor with actual size, and expand it in runtime (pad an activation tensor) without copy the whole tensor.
class ExpandablePhyBlock {
public:
  ExpandablePhyBlock(int device_id, size_t block_size);
  ~ExpandablePhyBlock();

  bool map_virtual_address(CUdeviceptr v_offset_addr, size_t size);

  bool unmap_virtual_address(CUdeviceptr v_offset_addr, size_t size);

  static std::atomic<int> thread_safe_counter;

  int block_id = -1;

  size_t block_size = 0;

  size_t remaining_size = 0;

  int device_id = 0;

  using Address = uintptr_t;
  std::map<Address, size_t> mapped_addresses;

  VmmAllocator* allocator = nullptr;

  BlockPool<ExpandablePhyBlock>* owned_pool = nullptr;

  CUmemGenericAllocationHandle alloc_handle;

  CUresult status;
};

static bool BlockComparator(const ExpandablePhyBlock* a, const ExpandablePhyBlock* b) {
  return (uintptr_t)a->block_id < (uintptr_t)b->block_id;
}

typedef bool (*Comparison)(const ExpandablePhyBlock*, const ExpandablePhyBlock*);

template<class Block>
struct BlockPool {};

template<>
struct BlockPool<ExpandablePhyBlock> {

    std::set<ExpandablePhyBlock*, Comparison> blocks;

    VmmAllocator* allocator = nullptr;

    BlockPool() : blocks(BlockComparator) {}

    bool add(ExpandablePhyBlock* block);

    bool remove(ExpandablePhyBlock* block);
};

template<class Block>
struct OwnedBlockPool {};

template<>
struct OwnedBlockPool<ExpandablePhyBlock> {

    using BlockId = int;
    std::map<BlockId, std::shared_ptr<ExpandablePhyBlock>> blocks;
    std::map<size_t, std::vector<ExpandablePhyBlock*>> open_blocks;

    VmmAllocator* allocator = nullptr;

    OwnedBlockPool() {}

    bool add(std::shared_ptr<ExpandablePhyBlock> block);

    bool remove(ExpandablePhyBlock* block);

    ExpandablePhyBlock* find_available(size_t size);

    void update(ExpandablePhyBlock* block, size_t previous_remaining_size);
};

} // namespace nvgpu
