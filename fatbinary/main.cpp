#include <cstdint>
#include <cstdio>

// declare the symbols that the linker will automatically export
// (the symbol name format: __start_ + section name / __stop_ + section name)
extern const uint8_t __start_mysection;
extern const uint8_t __stop_mysection;

// Here we will read the binary data from the section ".mysection"
// and print the content of the section
// so, in general case, we can use this method to read the binary data
// add copy code to the target device and execute it.
// This method can merge the binary data into the executable file
// and the users will can directly run the binary data without any other files.
// I think this is like the fatbinary in CUDA.
// we need to custom linker rules since the ".mysection" is a custom section.
// we can use the linker script to merge the binary data into the executable
// file. the linker script is a powerful tool to customize the link process. you
// can see the linker script in the "myld.ld" file.

int main() {
  // 1. calculate the start and end pointer of the section
  const uint8_t* start = &__start_mysection;
  const uint8_t* stop = &__stop_mysection;

  // 2. calculate the length of the section
  size_t length = static_cast<size_t>(stop - start);

  printf("The .mysection start address: %p\n", (void*)start);
  printf("The .mysection end   address: %p\n", (void*)stop);
  printf("The .mysection size: %zu bytes\n", length);

  // 3. print the content of the section (hex output)
  for (size_t i = 0; i < length; i++) {
    printf("%02X ", start[i]);
  }
  printf("\n");

  return 0;
}
