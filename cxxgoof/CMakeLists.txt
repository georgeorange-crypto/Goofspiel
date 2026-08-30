#
# Author: 陈子聪 (Chen Zicong)
# Date: 2026-08-30
# Purpose: Top-level build for the C++ speed-optimised Goofspiel environment
#          & exact Nash solver, exposed to Python via pybind11.
#
#          Targets:
#            goofspiel._core  — Python extension (vectorised env + Nash solver)
#            goof_tests        — GoogleTest-style catch2 binary (optional)
#
#          Typical usage (Windows with the MSVC vcpkg toolchain or conda):
#            pip install pybind11 cmake-build-extension
#            pip install -e .          (setup.py invokes this CMakeLists.txt)
#          OR manual:
#            cmake -S cxxgoof -B cxxbuild
#                -DPYTHON_EXECUTABLE=$(python -c 'import sys;print(sys.executable)')
#            cmake --build cxxbuild --config Release -j
#
project(cxxgoof LANGUAGES CXX)
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Output directory: drop .so / .pyd next to goofspiel package root --- #
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../goofspiel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../goofspiel)

# --- Dependencies --- #
# pybind11: either installed via pip (pybind11.get_include()) or system-wide
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RC
    )
    if(NOT RC EQUAL 0)
        message(FATAL_ERROR
            "pybind11 not found. Run: pip install pybind11 "
            "or set -DPYTHON_EXECUTABLE=/path/to/python with pybind11 installed.")
    endif()
    find_package(pybind11 REQUIRED CONFIG PATHS ${PYBIND11_CMAKE_DIR})
endif()

# Eigen (header-only) — used for the small k×k LP matrix in Nash solver,
# and for Eigen::ArrayXi SIMD accelerated batch step over M environments.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 via config (version ${EIGEN3_VERSION_STRING})")
else()
    # Fall back to python-installed eigen (conda-forge: conda install eigen)
    # or a vendored copy under cxxgoof/third_party/eigen if present.
    set(_eigen_candidate "${PROJECT_SOURCE_DIR}/third_party/eigen")
    if(EXISTS "${_eigen_candidate}/Eigen/Core")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_eigen_candidate}")
        message(STATUS "Using vendored Eigen at ${_eigen_candidate}")
    else()
        message(WARNING "Eigen3 not found. Building without SIMD batch step.")
        add_library(Eigen3::Eigen INTERFACE IMPORTED) # empty fallback
    endif()
endif()

# HIGHS (optional) — if the user has libhighs + headers installed, we use the
# HiGHS simplex solver directly from C++.  Otherwise, we fall back via pybind
# to scipy.optimize.linprog (which also uses HiGHS under the hood).
option(CXXGOOF_USE_HIGHS "Use native HiGHS LP solver for Nash sub-problems" OFF)
if(CXXGOOF_USE_HIGHS)
    find_package(highs QUIET)
    if(highs_FOUND)
        message(STATUS "Found native HiGHS: Nash sub-LPs solved directly from C++")
        add_compile_definitions(CXXGOOF_HAVE_HIGHS=1)
    else()
        message(STATUS "CXXGOOF_USE_HIGHS=ON but highs not found — falling back to scipy LP via pybind callback")
    endif()
endif()

# --- Compiler options (speed-oriented) --- #
if(MSVC)
    add_compile_options(/W3 /O2 /EHsc /fp:precise /Zc:__cplusplus /Zc:inline)
    # Disable harmless noise warnings
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -fno-math-errno)
endif()

# --- Extension target: goofspiel._core --- #
pybind11_add_module(_core
    MODULE
    include/goofspiel/goof_env.h
    include/goofspiel/goof_nash.h
    include/goofspiel/goof_estimate.h
    src/goof_env.cc
    src/goof_nash.cc
    src/bindings.cc
)
target_include_directories(_core PRIVATE include)
target_link_libraries(_core PRIVATE Eigen3::Eigen)
if(highs_FOUND AND CXXGOOF_USE_HIGHS)
    target_link_libraries(_core PRIVATE highs::highs)
endif()
set_target_properties(_core PROPERTIES
    OUTPUT_NAME _core
    # Place the .pyd / .so in goofspiel/_core.* so Python sees goofspiel._core
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../goofspiel
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../goofspiel
)
# MSVC puts Release/Debug under a subdirectory in multi-config generators;
# override that so the pyd ends up directly next to goofspiel/__init__.py.
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${_cfg} _CFG)
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${PROJECT_SOURCE_DIR}/../goofspiel"
        RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${PROJECT_SOURCE_DIR}/../goofspiel"
        ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${PROJECT_BINARY_DIR}/lib"
    )
endforeach()

# --- Optional native C++ catch2 test binary --- #
option(CXXGOOF_BUILD_TESTS "Build native C++ catch2 test binary (goof_tests)" OFF)
if(CXXGOOF_BUILD_TESTS)
    find_package(Catch2 3 QUIET)
    if(Catch2_FOUND)
        add_executable(goof_tests tests/env_tests.cc tests/nash_tests.cc
                       src/goof_env.cc src/goof_nash.cc)
        target_include_directories(goof_tests PRIVATE include)
        target_link_libraries(goof_tests PRIVATE Eigen3::Eigen Catch2::Catch2WithMain)
        if(highs_FOUND AND CXXGOOF_USE_HIGHS)
            target_link_libraries(goof_tests PRIVATE highs::highs)
        endif()
        enable_testing()
        include(Catch)
        catch_discover_tests(goof_tests)
    else()
        message(STATUS "Catch2 not installed; skipping goof_tests binary.")
    endif()
endif()

message(STATUS "")
message(STATUS "=== cxxgoof configuration summary ===")
message(STATUS "  Target extension    : goofspiel._core (pybind11)")
message(STATUS "  C++ standard        : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler            : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Eigen SIMD          : $<IF:$<TARGET_EXISTS:Eigen3::Eigen>,YES,NO>")
message(STATUS "  Native HiGHS LP     : $<BOOL:${highs_FOUND}>")
message(STATUS "  Install C++ tests   : ${CXXGOOF_BUILD_TESTS}")
message(STATUS "=====================================")
message(STATUS "")
