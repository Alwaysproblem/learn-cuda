include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE GPU_SOURCES "*.cu")
file(GLOB_RECURSE CPU_SOURCES "*.c" "*.cc" "*.cpp")

add_executable(grid_and_block ${GPU_SOURCES} ${CPU_SOURCES})
target_link_libraries(grid_and_block PRIVATE ${CUDA_LIBRARIES})
target_compile_options(
  grid_and_block PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
                        $<$<COMPILE_LANGUAGE:CXX>:>)
