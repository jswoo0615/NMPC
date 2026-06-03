cmake_minimum_required(VERSION 3.14)
project(NMPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
# Jetson Nano 등에서 nvcc가 PATH에 없을 경우를 대비해 존재하는지 확인 후 세팅
if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
# nvcc가 내부적으로 호출할 호스트 컴파일러를 현재 CXX 컴파일러(g++-8)로 강제 동기화
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
check_language(CUDA)

add_executable(test_matrix_core test/test_matrix_core.cpp)
target_include_directories(test_matrix_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(test_dynamics test/test_dynamics.cpp)
target_include_directories(test_dynamics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(test_solver test/test_solver.cpp)
target_include_directories(test_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_executable(test_NMPC test/test_NMPC.cpp)
target_include_directories(test_NMPC PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_link_libraries(test_matrix_core gtest_main)
target_link_libraries(test_dynamics gtest_main)
target_link_libraries(test_solver gtest_main)
target_link_libraries(test_NMPC gtest_main)