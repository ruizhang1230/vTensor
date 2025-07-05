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

#include "common.h"
#include "cu_util.h"

#include <cuda.h>

#include <cuda_runtime_api.h>

enum class AllocatorType {
    VMM_ALOC = 0,
    N
};

/**
 * Description : Torch torch.cuda.memory.CUDAPluggableAllocator API see https://docs.pytorch.org/docs/stable/notes/cuda.html
 * The API facilitates replacing torch native allocator of torch.Tensor on request
 */
struct DeviceAllocatorBase {

    HOST DeviceAllocatorBase() {};

    virtual ~DeviceAllocatorBase() {}

    virtual HOST_INLINE void* alloc(size_t size, int device, CUstream stream) = 0;

    virtual HOST_INLINE void dealloc(void* ptr, size_t size, int device, CUstream stream) = 0;
};

void ensure_context(unsigned long long device);
