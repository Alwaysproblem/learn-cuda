include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE GPU_SOURCES "*.cu")
file(GLOB_RECURSE CPU_SOURCES "*.c" "*.cc" "*.cpp")

add_executable(warp_divergence ${GPU_SOURCES} ${CPU_SOURCES})
target_link_libraries(warp_divergence PRIVATE ${CUDA_LIBRARIES})
target_compile_options(
  warp_divergence PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
                         $<$<COMPILE_LANGUAGE:CXX>:>)
