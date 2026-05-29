cmake_minimum_required(VERSION 3.15)
project(kokoro-infer LANGUAGES CXX)

option(USE_RKNN "Enable RKNN decoder backend (RK3588 NPU)" OFF)
option(USE_ORT_CUDA "Enable ONNX Runtime CUDA/TensorRT EPs" OFF)
option(BUILD_SERVER "Build kokoro-server (needs drogon + opus)" ON)
option(BUILD_CLI "Build kokoro-cli" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# ---- ONNX Runtime --------------------------------------------------------
# Two layouts supported:
#   1. ORT_ROOT=/path/to/onnxruntime-linux-*-1.14.x  (paroli's flat layout)
#   2. Unset → use system install. Arch's onnxruntime-cpu puts headers under
#      /usr/include/onnxruntime/, which is what the cxx_api expects.
set(ORT_ROOT "" CACHE STRING "Path to ONNX Runtime root (optional, else system)")
if(ORT_ROOT STREQUAL "")
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES onnxruntime)
  if(NOT ORT_INCLUDE_DIR)
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found; set -DORT_ROOT=...")
  endif()
  include_directories(${ORT_INCLUDE_DIR})
else()
  include_directories(${ORT_ROOT}/include)
  link_directories(${ORT_ROOT}/lib)
endif()

# ---- espeak-ng ----------------------------------------------------------
# Either ESPEAK_NG_ROOT=/install/prefix (with include/espeak-ng/ and lib/) or
# system install. Used by src/phonemizer.cpp directly — we don't wrap
# piper-phonemize since the code only touches the espeak-ng C API.
set(ESPEAK_NG_ROOT "" CACHE STRING "Path to espeak-ng install prefix (optional)")
if(NOT ESPEAK_NG_ROOT STREQUAL "")
  include_directories(${ESPEAK_NG_ROOT}/include)
  link_directories(${ESPEAK_NG_ROOT}/lib)
endif()

# ---- BLAS for iSTFT sgemm ------------------------------------------------
# Try cblas (openblas/atlas) via the standard finder.
find_package(BLAS REQUIRED)
# cblas.h location varies; try common spots.
find_path(CBLAS_INCLUDE_DIR cblas.h
  PATH_SUFFIXES openblas blis)
if(NOT CBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "cblas.h not found (install libopenblas-dev or similar)")
endif()
include_directories(${CBLAS_INCLUDE_DIR})

# ---- Other deps ----------------------------------------------------------
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# ---- misaki-cpp (English G2P front end) ---------------------------------
# The faithful C++ port of misaki/en.py — replaces the old in-tree English
# pipeline (src/lexicon.cpp + src/num2words_en.cpp are now unused). It ships its
# own data (dictionaries, perceptron tagger, spaCy contraction exceptions) under
# misaki-cpp/data, which is staged next to the binaries as "misaki-data".
# Keep its espeak in sync with ours so a single libespeak-ng is linked.
if(NOT ESPEAK_NG_ROOT STREQUAL "")
  set(ESPEAK_ROOT "${ESPEAK_NG_ROOT}" CACHE STRING "espeak-ng install prefix" FORCE)
endif()
set(MISAKI_BUILD_TOOLS OFF CACHE BOOL "" FORCE)  # no misaki-cli/test in this build
add_subdirectory(misaki-cpp)
set(MISAKI_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/misaki-cpp/data")

# ============== libkokoro ================================================
add_library(kokoro
  src/istft.cpp
  src/phonemizer.cpp
  src/g2p.cpp
  src/kokoro.cpp
  src/onnx-decoder.cpp)

target_include_directories(kokoro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kokoro
  PUBLIC
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
  PRIVATE
    misaki
    onnxruntime
    espeak-ng
    cblas
    ${BLAS_LIBRARIES}
    pthread
    dl)

if(USE_RKNN)
  target_sources(kokoro PRIVATE src/rknn-decoder.cpp)
  target_compile_definitions(kokoro PRIVATE USE_RKNN)
  target_link_libraries(kokoro PRIVATE rknnrt)
endif()

if(USE_ORT_CUDA)
  target_compile_definitions(kokoro PRIVATE USE_ORT_CUDA)
endif()

# Copy the misaki data dir next to whichever binaries get built, so the
# default --lexicon-dir (exe_dir/misaki-data) just works.
function(stage_misaki_data tgt)
  add_custom_command(TARGET ${tgt} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${MISAKI_DATA_DIR}" "$<TARGET_FILE_DIR:${tgt}>/misaki-data"
    COMMENT "Staging misaki lexicon next to ${tgt}")
endfunction()

# ============== kokoro-cli ===============================================
if(BUILD_CLI)
  add_executable(kokoro-cli cli/main.cpp)
  target_link_libraries(kokoro-cli PRIVATE kokoro)
  stage_misaki_data(kokoro-cli)
endif()

# ============== kokoro-server ============================================
if(BUILD_SERVER)
  find_package(Drogon REQUIRED)
  find_package(Opus REQUIRED)

  add_executable(kokoro-server
    server/main.cpp
    server/api.cpp
    server/OggOpusEncoder.cpp)

  target_include_directories(kokoro-server PRIVATE ${OPUS_INCLUDE_DIRS})
  target_link_libraries(kokoro-server PRIVATE
    kokoro
    Drogon::Drogon
    soxr
    ${OPUS_LIBRARIES}
    opusenc
    ogg)
  target_precompile_headers(kokoro-server PRIVATE server/pch.hpp)
  stage_misaki_data(kokoro-server)
endif()
