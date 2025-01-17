#include <cstdint>

__attribute__((section(".mysection"),
               used)) static const uint8_t g_myBinaryData[] = {
    // Here asume that we have some binary data
    // The data could be anything, for example, ptx code
    // or some other binary data for specified target device.
    // could be elf, mach-o, or other binary format for other target device.
    0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};

// declare the size of the binary data
__attribute__((section(".mysection"),
               used)) static const uint64_t g_myBinarySize =
    sizeof(g_myBinaryData);
