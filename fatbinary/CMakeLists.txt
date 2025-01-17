file(GLOB_RECURSE CPU_SOURCES "*.c" "*.cc" "*.cpp")

add_executable(fatbinary ${CPU_SOURCES})
set(LD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/myld.ld")
target_link_options(fatbinary PRIVATE -Wl,-T,${LD_SCRIPT})
