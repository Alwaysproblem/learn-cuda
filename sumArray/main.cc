#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <iostream>

#include "device.h"

void printVector(float *vector, const int N) {
  for (int i = 0; i < N; i++) {
    printf("%f ", vector[i]);
  }
  printf("\n");
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++) {
    C[idx] = A[idx] + B[idx];
  }
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
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }

  if (match) printf("Arrays match.\n\n");

  return;
}

int main(int argc, char **argv) {
  int nElem = 1024;
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *h_C, *h_C_gpu;
  const int grid_size = 10;
  const int block_size = 256;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);
  h_C_gpu = (float *)malloc(nBytes);

  initialData(h_A, nElem);
  initialData(h_B, nElem);

  auto start = std::chrono::high_resolution_clock::now();
  sumArraysOnHost(h_A, h_B, h_C, nElem);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time on Host: " << elapsed.count() * 1000 << " ms\n";

  auto start_gpu = std::chrono::high_resolution_clock::now();
  sumArrayOnGPU(h_A, h_B, h_C_gpu, nElem, grid_size, block_size);
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
  std::cout << "Elapsed time: " << elapsed_gpu.count() * 1000 << " ms\n";

  printVector(h_C, nElem);
  printf("------------------\n");
  printVector(h_C_gpu, nElem);
  printf("------------------\n");
  checkResult(h_C, h_C_gpu, nElem);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_gpu);

  cudaDeviceReset();
  return (0);
}
