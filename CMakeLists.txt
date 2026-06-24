cmake_minimum_required(VERSION 3.14)
project(nmpc_core_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==============================================================================
# [Dependency Resolution] 파이썬 C-API 호환성을 위한 최신 Pybind11 동적 Fetch
# ==============================================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.1 # Python 3.10 이상을 완벽 지원하는 안정화 버전
)
FetchContent_MakeAvailable(pybind11)

# ==============================================================================
# [Include Directory Setup]
# ==============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ==============================================================================
# [Target Definition]
# ==============================================================================
# 래퍼 소스 코드가 src 디렉토리 하위에 위치해야 함을 잊지 마십시오.
pybind11_add_module(nmpc_core src/nmpc_wrapper.cpp)

# ==============================================================================
# [Compiler Flags Tuning] 하드 리얼타임을 위한 극한의 최적화
# ==============================================================================
target_compile_options(nmpc_core PRIVATE 
    -O3 
    -march=native 
    -ffast-math 
    -funroll-loops
    -fomit-frame-pointer
    -Wno-psabi        # 64-byte alignment 관련 경고 무시
    -Wno-sign-compare # FastMath의 index 비교 경고 무시
)

# Interprocedural Optimization (LTO)
# 링크 단계에서 함수 호출 오버헤드를 완전히 박살내기 위한 인라인화
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set_target_properties(nmpc_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_include_directories(nmpc_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)