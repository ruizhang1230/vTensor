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

// TOOD (yiakwy) : replace to fmtlib
#include <iostream>
#include <cassert>

#define LOGE(format, ...)                                                      \
  fprintf(stdout, "L%d:" format "\n", __LINE__, ##__VA_ARGS__);                \
  fflush(stdout);

#define ASSERT(cond, ...)                                                      \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGE(__VA_ARGS__);                                                       \
      assert(0);                                                               \
    }                                                                          \
  }
#define WARN(cond, ...)                                                        \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGE(__VA_ARGS__);                                                       \
    }                                                                          \
  }
