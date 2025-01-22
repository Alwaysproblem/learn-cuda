#include <__clang_cuda_runtime_wrapper.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <iostream>

#include "device.h"

void printMatrix(float *C, const int nx, const int ny) {
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      printf("%f ", ic[ix]);
    }

    ic += nx;
    printf("\n");
  }

  return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }

    ia += nx;
    ib += nx;
    ic += nx;
  }

  return;
}

void initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
      break;
    }
  }

  if (match)
    printf("Arrays match.\n\n");
  else
    printf("Arrays do not match.\n\n");
}

int main(int argc, char **argv) {
  // set up data size of matrix
  int nx = 50;
  int ny = 50;

  int nElem = nx * ny;
  int nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *h_C, *h_C_gpu;

  dim3 block_size(32, 32);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x,
                 (ny + block_size.y - 1) / block_size.y);

  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);
  h_C_gpu = (float *)malloc(nBytes);

  initialData(h_A, nElem);
  initialData(h_B, nElem);

  auto start = std::chrono::high_resolution_clock::now();
  sumMatrixOnHost(h_A, h_B, h_C, nx, ny);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time on Host: " << elapsed.count() * 1000 << " ms\n";

  auto start_gpu = std::chrono::high_resolution_clock::now();
  sumMatrixOnGPU(h_A, h_B, h_C_gpu, nx, ny, grid_size, block_size);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
  std::cout << "Elapsed time: " << elapsed_gpu.count() * 1000 << " ms\n";

  printMatrix(h_C, nx, ny);
  printf("------------------\n");
  printMatrix(h_C_gpu, nx, ny);
  printf("------------------\n");
  checkResult(h_C, h_C_gpu, nElem);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_gpu);

  cudaDeviceReset();
  return (0);
}
