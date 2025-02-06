#include "device.h"

#define MATHKERNEL(name)                                                       \
  extern "C" void name(float *c, int N, const dim3 grid, const dim3 block) {   \
    float *d_C;                                                                \
    cudaMalloc((float **)&d_C, N * sizeof(float));                             \
    name##OnGPU<<<block, grid>>>(d_C);                                         \
    cudaMemcpy(c, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);             \
    cudaFree(d_C);                                                             \
  }

__global__ void mathKernel1OnGPU(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if (tid % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }

  c[tid] = ia + ib;
}

__global__ void mathKernel2OnGPU(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }

  c[tid] = ia + ib;
}

__global__ void mathKernel3OnGPU(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  bool ipred = (tid % 2 == 0);

  if (ipred) {
    ia = 100.0f;
  }

  if (!ipred) {
    ib = 200.0f;
  }

  c[tid] = ia + ib;
}

__global__ void mathKernel4OnGPU(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  int itid = tid >> 5;

  if (itid & 0x01 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }

  c[tid] = ia + ib;
}

__global__ void warmingupOnGPU(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    ia = 100.0f;
  } else {
    ib = 200.0f;
  }

  c[tid] = ia + ib;
}

MATHKERNEL(mathKernel1);
MATHKERNEL(mathKernel2);
MATHKERNEL(mathKernel3);
MATHKERNEL(mathKernel4);
MATHKERNEL(warmingup);
