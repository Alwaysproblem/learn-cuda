#include "device.h"

__global__ void sumArray(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// 在 device.cu 中封装函数调用
extern "C" void sumArrayOnGPU(float *A, float *B, float *C, const int N,
                              const int grid_size, const int block_size) {
  dim3 block(grid_size);
  dim3 grid(block_size);
  float *d_A, *d_B, *d_C;
  cudaMalloc((float **)&d_A, N * sizeof(float));
  cudaMalloc((float **)&d_B, N * sizeof(float));
  cudaMalloc((float **)&d_C, N * sizeof(float));
  cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
  sumArray<<<block, grid>>>(d_A, d_B, d_C, N);
  cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
