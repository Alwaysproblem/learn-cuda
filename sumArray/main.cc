#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

  sumArraysOnHost(h_A, h_B, h_C, nElem);

  sumArrayOnGPU(h_A, h_B, h_C_gpu, nElem, grid_size, block_size);

  printVector(h_C, nElem);
  printf("\n");
  printVector(h_C_gpu, nElem);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_gpu);

  return (0);
}
