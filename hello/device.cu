/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

#include "device.h"

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

// 在 device.cu 中封装函数调用
extern "C" void launch_helloFromGPU() {
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
}
