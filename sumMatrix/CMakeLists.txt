include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE GPU_SOURCES "*.cu")
file(GLOB_RECURSE CPU_SOURCES "*.c" "*.cc" "*.cpp")

add_executable(sum_matrix ${GPU_SOURCES} ${CPU_SOURCES})
target_link_libraries(sum_matrix PRIVATE ${CUDA_LIBRARIES})
target_compile_options(sum_matrix PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>
                                         $<$<COMPILE_LANGUAGE:CXX>:>)
