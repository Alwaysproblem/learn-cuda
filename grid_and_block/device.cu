#include <stdio.h>
#include <cstdio>
#include "device.h"

__global__ void checkIndex_imp(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("--------------------------------\n");
    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

// 在 device.cu 中封装函数调用
extern "C" void checkIndex(dim3 grid, dim3 block) {
    checkIndex_imp<<<grid, block>>>();
    cudaDeviceReset();
}
