#include "allocator/allocator.h"

// Adpated from https://github.com/vllm-project/vllm/pull/11743
void ensure_context(unsigned long long device) {
  CUcontext pctx;
  DRV_CALL(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    DRV_CALL(cuDevicePrimaryCtxRetain(&pctx, device));
    DRV_CALL(cuCtxSetCurrent(pctx));
  }
}
