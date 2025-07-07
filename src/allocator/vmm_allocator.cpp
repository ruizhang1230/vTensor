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

#include "cu_util.h"

#include "allocator/vmm_allocator.h"

#include <iostream>

// MACRO better to be in cpp files
// #define ROUND_UP(x, n) (((x) + ((n) - 1)) / (n) * (n))

#define CEIL_DIV(x, n) (((x) + (n) - 1) /  (n))
#define ROUND_UP(x, n) (CEIL_DIV(x, n) * (n))

namespace nvgpu {
    // This enables creating torch tensor device memory with VMM API
    HOST_INLINE void* VmmAllocator::alloc(size_t size, int device, CUstream stream) {
        ensure_context(device);

        CUdeviceptr dptr;
        size_t reserved_size;
        DRV_CALL(reserve_virtual_addr((void**)&dptr, size, &reserved_size, device, stream));

        // find the nearest memory block
        PhyBlock* block = owned_pool.find_available(size);

        std::shared_ptr<PhyBlock> _block = nullptr;
        if (block == nullptr) {
            _block = std::make_shared<PhyBlock>(device, reserved_size);
            block = _block.get();
        }

        // mapping virtual addr to the device memory
        map_virtual_address(block, (void *)dptr, reserved_size);

        if (_block != nullptr) {
            bool status = owned_pool.add(_block);
            assert(status);
        }
        return (void *)dptr;
    }

    // Adpated from vTensor original impl and [vllm](https://github.com/vllm-project/vllm/pull/11743), used for vTensor internal alloc
    HOST_INLINE CUresult VmmAllocator::reserve_virtual_addr(void** ptr, size_t request_size, size_t* reserved_size, int device, CUstream stream) {
        ensure_context(device);

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

        size_t granularity;
        DRV_CALL(cuMemGetAllocationGranularity(&granularity, &prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        *reserved_size = ROUND_UP(request_size, granularity);

        CUdeviceptr v_ptr;
        DRV_CALL(cuMemAddressReserve(&v_ptr, *reserved_size, 0ULL/*alignment*/, 0ULL/*extension addr offset*/, 0ULL/**/));

        *ptr = (void *)v_ptr;

        return CUDA_SUCCESS;
    }

    HOST_INLINE void VmmAllocator::dealloc(void* ptr/*virtual_memroy_address*/, size_t size, int device, CUstream stream) {
        ensure_context(device);

        PhyBlock* block = get_allocated_block(ptr);

        if (block != nullptr) {
            unmap_virtual_address(block, ptr, size);
        }

        /*
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
        if (block->unmap_virtual_address(dptr, size) ) {
            unmap_virtual_address(block->device_id, size, dptr);
        }
         */
    }

    HOST_INLINE void VmmAllocator::map_virtual_address(VmmAllocator::PhyBlock* block, void* v_offset_addr, size_t size) {
        assert(block != nullptr);

        block->map_virtual_address(reinterpret_cast<CUdeviceptr>(v_offset_addr), size);

        auto inserted = allocated_blocks.insert({reinterpret_cast<uintptr_t>(v_offset_addr), block});

        if (inserted.second) {
            std::cout << "[VmmAllocator::map_virtual_address] add mapping of <block#" << block->block_id << ", " << (uintptr_t)v_offset_addr << ", " << size << ">" << std::endl;
        } else {
            std::cout << "[VmmAllocator::map_virtual_address] failed to add mapping of <block#" << block->block_id << ", " << (uintptr_t)v_offset_addr << ", " << size << ">" << std::endl;
        }

    }

    HOST_INLINE void VmmAllocator::unmap_virtual_address(PhyBlock* block, void *v_offset_addr, size_t size) {
        ensure_context(block->device_id);
        /*
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
        if (block->unmap_virtual_address(dptr, size) ) {
            unmap_virtual_address(block->device_id, size, dptr);
        }
         */
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(v_offset_addr);

        size_t old_capacity = block->remaining_size;
        if (block->unmap_virtual_address(dptr, size) ) {
            owned_pool.update(block, old_capacity);
        }
    }

    /*
    HOST_INLINE void VmmAllocator::unmap_virtual_address(int device, size_t size, CUdeviceptr dptr) {
        ensure_context(device);

        DRV_CALL(cuMemUnmap(dptr, size));
        DRV_CALL(cuMemAddressFree(dptr, size));
    }
    */

    HOST_INLINE VmmAllocator::PhyBlock* VmmAllocator::get_allocated_block(void* ptr, bool remove) {
        // TODO (yiakwy) : add mutex shards to enable fine control of concurrent accesses
        std::lock_guard<std::mutex> lock(mtx);
        auto it = allocated_blocks.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == allocated_blocks.end()) {
            std::cout << "[VmmAllocator::get_allocated_block] cannot find a block associated  to virtual address " << (uintptr_t)ptr << " " << std::endl;
            return nullptr;
        }
        PhyBlock* block = it->second;
        std::cout << "[VmmAllocator::get_allocated_block] find block#" << block->block_id <<  " associated to virtual address " << (uintptr_t)ptr << " " << std::endl;
        if (remove) {
            allocated_blocks.erase(it);
            if (block->owned_pool != nullptr) {
                block->owned_pool->remove(block);
            } else {
                owned_pool.remove(block);
            }

            std::cout << "[VmmAllocator::get_allocated_block] remove mapping of <block#" << block->block_id << ", " << (uintptr_t)ptr << ">" << std::endl;
        }
        return block;
    }

} // namespace nvgpu
