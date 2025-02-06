#include <__clang_cuda_runtime_wrapper.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <iostream>

#include "device.h"

void timeit(const char *msg, void (*fn)(float *, int, const dim3, const dim3),
            float *C, int size, dim3 grid, dim3 block) {
  auto start = std::chrono::high_resolution_clock::now();
  fn(C, size, grid, block);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << msg << " " << elapsed.count() * 1000 << " ms\n";
}

int main(int argc, char **argv) {
  // set up data size of matrix
  int size = 64;
  int blocksize = 64;
  if (argc > 1) blocksize = atoi(argv[1]);

  if (argc > 2) size = atoi(argv[2]);
  printf("Data size %d ", size);

  // set up execution configuration
  dim3 block(blocksize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);

  float *C;
  size_t nBytes = size * sizeof(float);
  C = (float *)malloc(nBytes);
  for (int i = 0; i < 5; i++) warmingup(C, size, grid, block);

  timeit("mathKernel1", mathKernel1, C, size, grid, block);
  timeit("mathKernel2", mathKernel2, C, size, grid, block);
  timeit("mathKernel3", mathKernel3, C, size, grid, block);
  timeit("mathKernel4", mathKernel4, C, size, grid, block);

  free(C);

  cudaDeviceReset();
  return (0);
}
