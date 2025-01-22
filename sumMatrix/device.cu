#include "device.h"

__global__ void sumMatrix(float *A, float *B, float *C, const int nx, const int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = i * ny + j;
  if (i < nx && j < ny) {
    C[idx] = A[idx] + B[idx];
  }
}

// 在 device.cu 中封装函数调用
extern "C" void sumMatrixOnGPU(float *A, float *B, float *C, const int nx,
                               const int ny, const dim3 grid,
                               const dim3 block) {
  int N = nx * ny;
  float *d_A, *d_B, *d_C;
  cudaMalloc((float **)&d_A, N * sizeof(float));
  cudaMalloc((float **)&d_B, N * sizeof(float));
  cudaMalloc((float **)&d_C, N * sizeof(float));
  cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
  sumMatrix<<<block, grid>>>(d_A, d_B, d_C, nx, ny);
  cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
