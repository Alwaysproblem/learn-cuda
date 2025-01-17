#include <stdio.h>

#include "device.h"

int main(int argc, char **argv) {
  printf("Hello World from CPU!\n");

  launch_helloFromGPU();
  return 0;
}
