#include <stdio.h>

#include <cstdio>

#include "device.h"

__global__ void checkIndex_imp(void) {
  printf(
      "threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
      "gridDim:(%d, %d, %d)\n",
      threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
      blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

// 在 device.cu 中封装函数调用
extern "C" void checkIndex(dim3 grid, dim3 block) {
  checkIndex_imp<<<grid, block>>>();
  cudaDeviceReset();
}
