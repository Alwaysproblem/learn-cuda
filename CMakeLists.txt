cmake_minimum_required(VERSION 3.18)

project(cuda-tutorial LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Add a target to generate cuda executable
function(add_cuda_executable target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE ${CUDA_LIBRARIES})
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
  target_compile_options(${target}
                         PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler>)
endfunction(add_cuda_executable target)

# set(DCMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")

# other third party libraries find_package(OpenMP) find_package(OpenGL)
# find_package(GLUT)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

add_subdirectory(hello)
add_subdirectory(fatbinary)
add_subdirectory(sumArray)
add_subdirectory(sumMatrix)
add_subdirectory(device_info)
add_subdirectory(grid_and_block)
add_subdirectory(warp_divergence)

# -------------------------------------------------------------------#
# Configuration summary
# -------------------------------------------------------------------#
message("//===================================================")
message("// ${PROJECT_NAME} build configuration:")
message("// ${PROJECT_NAME} version : ${PROJECT_VERSION}")
message("//===================================================")
message("  CMake version          : ${CMAKE_VERSION}")
if(NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION} " "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CXX      STANDARD  : ${CMAKE_CXX_STANDARD}")
message("  CUDA CXX STANDARD  : ${CMAKE_CUDA_STANDARD}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("")
# message(" Thrid party libraries") message("  OpenGL found : ${OPENGL_FOUND}")
