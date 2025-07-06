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
#include <algorithm>

#include "cu_util.h"

#include "allocator/expandable_phyblock.h"

#define CEIL_DIV(x, n) (((x) + (n) - 1) / (n))
#define ROUND_UP(x, n) (CEIL_DIV(x, n) * (n))

namespace nvgpu {

    std::atomic<int> ExpandablePhyBlock::thread_safe_counter{0};

    ExpandablePhyBlock::ExpandablePhyBlock(int device_id, size_t block_size) {
        this->device_id = device_id;

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;

        size_t granularity;
        DRV_CALL(cuMemGetAllocationGranularity(&granularity, &prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        int aligned_block_size = ROUND_UP(block_size, granularity);

        this->block_size = aligned_block_size;
        this->remaining_size = aligned_block_size;

        status = cuMemCreate(&alloc_handle, aligned_block_size, &prop, 0ULL);

        this->block_id = thread_safe_counter++;
    }

    ExpandablePhyBlock::~ExpandablePhyBlock() {
        std::cout << "[ExpandablePhyBlock::~ExpandablePhyBlock] [Block#" << block_id << "]" << " deallocating device memory ..." << std::endl;
        if (status == CUDA_SUCCESS) {
            status = cuMemRelease(alloc_handle);
            if (status != CUDA_SUCCESS) {
                std::cout << "[ExpandablePhyBlock::~ExpandablePhyBlock] [Block#" << block_id << "]" << " failed to deallocate device memory ..." << std::endl;
            } else {
                std::cout << "[ExpandablePhyBlock::~ExpandablePhyBlock] [Block#" << block_id << "]" << " device memory deallocated." << std::endl;
            }
            // DRV_CALL(status);
        }
    }

    bool ExpandablePhyBlock::map_virtual_address(CUdeviceptr v_offset_addr, size_t size) {
        if (remaining_size >= size) {
            auto addr_inserted = mapped_addresses.insert({reinterpret_cast<uintptr_t>((void *)v_offset_addr), size});
            if (!addr_inserted.second) {
                std::cout << "[ExpandablePhyBlock::map_virtual_address] [Block#" << block_id << "] failed to map addresss, remaining size " << remaining_size << "." << std::endl;
                return false;
            }

            DRV_CALL(cuMemMap(v_offset_addr, size, 0ULL, alloc_handle, 0ULL));

            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = this->device_id;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            DRV_CALL(cuMemSetAccess(v_offset_addr, size, &accessDesc, 1));

            remaining_size -= size;

            std::cout << "[ExpandablePhyBlock::map_virtual_address] [Block#" << block_id << "] mapping address successufully, remaining size " << remaining_size << "." << std::endl;
            return true;
        }
        std::cout << "[ExpandablePhyBlock::map_virtual_address] [Block#" << block_id << "] failed to map addresss, remaining size " << remaining_size << "." << std::endl;
        return false;
    }

    bool ExpandablePhyBlock::unmap_virtual_address(CUdeviceptr v_offset_addr, size_t size) {
        auto it = mapped_addresses.find(reinterpret_cast<uintptr_t>((void*)v_offset_addr));
        if (it != mapped_addresses.end()) {
            assert(it->second == size);

            mapped_addresses.erase(it);

            remaining_size += size;
            return true;
        }
        return false;
    }


    bool BlockPool<ExpandablePhyBlock>::add(ExpandablePhyBlock* block) {
        assert(block->owned_pool == nullptr);

        block->owned_pool = this;

        if (block->allocator != nullptr && block->allocator != this->allocator) {
            std::cout << "[BlockPool::add] Failed to add the block#" << block->block_id << " to blockPool" << std::endl;
            exit(0);
        } else {
            block->allocator = this->allocator;
        }

        auto inserted = blocks.insert(block);

        if (inserted.second) {
            std::cout << "[BlockPool::add] add Block#" << block->block_id << "." << std::endl;
        } else {
            std::cout << "[BlockPool::add] failed to add Block#" << block->block_id << "." << std::endl;
        }

        return inserted.second;
    }

    bool BlockPool<ExpandablePhyBlock>::remove(ExpandablePhyBlock* block) {
        int removed = blocks.erase(block);

        block->owned_pool = nullptr;

        if (removed > 0) {
            std::cout << "[BlockPool::remove] remove Block#" << block->block_id << std::endl;
        } else {
            std::cout << "[BlockPool::remove] failed to remove Block#" << block->block_id << std::endl;
            if (block->block_id == 2) {

            }
        }
        return removed > 0;
    }

    bool OwnedBlockPool<ExpandablePhyBlock>::add(std::shared_ptr<ExpandablePhyBlock> block) {
        if (block->allocator != nullptr && block->allocator != this->allocator) {
            std::cout << "[OwnedBlockPool::add] Failed to add the block#" << block->block_id << " to blockPool" << std::endl;
            exit(0);
        } else {
            block->allocator = this->allocator;
        }

        auto inserted = blocks.insert({block->block_id, block});
        if (!inserted.second) {
            std::cout << "[OwnedBlockPool::add] failed to add Block#" << block->block_id << "." << std::endl;
        } else {
            if (block->remaining_size > 0) {
                auto open_blocks_inserted = open_blocks.insert({block->remaining_size,{block.get()}});
                if (!open_blocks_inserted.second) {
                    open_blocks_inserted.first->second.push_back(block.get());
                    std::cout << "[OwnedBlockPool::add] Block#" << block->block_id << " is now available for allocating maximum " << block->remaining_size << " bytes memory." << std::endl;
                }
            }
            std::cout << "[OwnedBlockPool::add] add Block#" << block->block_id << "." << std::endl;
        }

        return inserted.second;
    }

    bool OwnedBlockPool<ExpandablePhyBlock>::remove(ExpandablePhyBlock* block) {
        auto it = blocks.find(block->block_id);
        if (it != blocks.end()) {
            assert(it->second.get() == block);

            auto pos = open_blocks.find(block->remaining_size);
            if (pos != open_blocks.end()) {
                auto & vec = pos->second;
                if (vec.size() == 1) {
                    vec.clear();
                } else {
                    vec.erase( std::find(vec.begin(), vec.end(), block) );
                }
            }

            blocks.erase(it);
            std::cout << "[OwnedBlockPool::remove] remove Block#" << block->block_id << std::endl;
            return true;
        } else {
            std::cout << "[OwnedBlockPool::remove] failed to remove Block#" << block->block_id << std::endl;
            return false;
        }
    }

    ExpandablePhyBlock* OwnedBlockPool<ExpandablePhyBlock>::find_available(size_t size) {
        auto it = open_blocks.upper_bound(size);
        if (it != open_blocks.end()) {
            if (it->second.size() > 0) {
                ExpandablePhyBlock* block = it->second.back();
                it->second.pop_back();

                if (block->remaining_size - size > 0) {
                    auto open_blocks_inserted = open_blocks.insert({block->remaining_size - size, {block}});
                    if (!open_blocks_inserted.second) {
                        open_blocks_inserted.first->second.push_back(block);
                        std::cout << "[OwnedBlockPool::add] Block#" << block->block_id << " will be available again for allocating maximum " << block->remaining_size - size << " bytes memory." << std::endl;
                    }
                }

                return block;
            }
        }
        return nullptr;
    }

} // namespace nvgpu
