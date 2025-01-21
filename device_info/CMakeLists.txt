include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE GPU_SOURCES "*.cu")
file(GLOB_RECURSE CPU_SOURCES "*.c" "*.cc" "*.cpp")

add_executable(print_device_info ${GPU_SOURCES} ${CPU_SOURCES})
target_link_libraries(print_device_info PRIVATE ${CUDA_LIBRARIES})
target_compile_options(
  print_device_info PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
                           $<$<COMPILE_LANGUAGE:CXX>:>)
