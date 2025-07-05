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

#include <map>
#include <memory>
#include <mutex>

#include <vector>

#include "allocator.h"
#include "expandable_phyblock.h"

namespace nvgpu {

struct VmmAllocator : public DeviceAllocatorBase {

    using PhyBlock = ExpandablePhyBlock;
    using Ptr = std::shared_ptr<VmmAllocator>;

    BlockPool<PhyBlock> shared_pool;

    BlockPool<PhyBlock> exclusive_pool;

    OwnedBlockPool<ExpandablePhyBlock> owned_pool;

    // TODO (yiakwy) : add mutex shards to enable fine control of concurrent accesses
    std::mutex mtx;

    // TODO (yiakwy) : add binary tree to retrieve blocks, use flat_map later
    using Address = uintptr_t;
    std::map<Address, PhyBlock*> allocated_blocks;

    HOST VmmAllocator() : DeviceAllocatorBase() {
        shared_pool.allocator = this;
        exclusive_pool.allocator = this;
    }

    HOST virtual ~VmmAllocator() {}

    HOST static VmmAllocator::Ptr instance() {
        static Ptr instance(new VmmAllocator());
        return instance;
    }

    // Pytorch allocator API

    virtual HOST_INLINE void* alloc(size_t size, int device, CUstream stream) override ;

    virtual HOST_INLINE void dealloc(void* ptr, size_t size, int device, CUstream stream) override ;

    // VMM reserve virtual addresses API

    HOST_INLINE CUresult reserve_virtual_addr(void** ptr/*dest*/, size_t request_size, size_t* reserved_size, int device, CUstream stream);

    // VMM mapping virtual addresses API

    HOST_INLINE void map_virtual_address(CUmemAllocationProp prop, CUdeviceptr dptr, size_t size, CUmemGenericAllocationHandle* alloc_handle);

    HOST_INLINE void map_virtual_address(PhyBlock* block, void* v_offset_addr, size_t size);

    // VMM unmapping virtual addresses API

    HOST_INLINE void unmap_virtual_address(int device, size_t size, CUdeviceptr dptr);

    HOST_INLINE PhyBlock* get_allocated_block(void* ptr, bool remove = false);
};

} // namepsace nvgpu
