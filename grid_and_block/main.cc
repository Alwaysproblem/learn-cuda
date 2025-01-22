#include <math.h>

#include <iostream>

#include "device.h"

int main(int argc, char **argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return 1;
  }

  dim3 block(3);
  dim3 grid(2);

  checkIndex(grid, block);

  cudaDeviceReset();
  return 0;
}
