# LibTorch + GPU + CMake < 3.31:
# start_fall_detection.sh auto-detects GPU arch from nvidia-smi (TORCH_CUDA_ARCH_LIST).
# For manual cmake: export TORCH_CUDA_ARCH_LIST=8.6  (sm_75→7.5  sm_80→8.0  sm_86→8.6  sm_89→8.9)
cmake_minimum_required(VERSION 3.20)
project(fall_detection_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Forward arch list into env so PyTorch skips CUDA detection on CMake < 3.31.
set(FALL_TORCH_CUDA_ARCH_LIST "" CACHE STRING
    "GPU arch list, e.g. 8.6  (sm_75→7.5  sm_80→8.0  sm_86→8.6  sm_89→8.9)")
if(FALL_TORCH_CUDA_ARCH_LIST)
    set(ENV{TORCH_CUDA_ARCH_LIST} "${FALL_TORCH_CUDA_ARCH_LIST}")
endif()

# ── Dependencies ──────────────────────────────────────────────
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn)
find_package(CURL CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(redis++ CONFIG REQUIRED)
find_package(RdKafka CONFIG REQUIRED)
find_package(aws-cpp-sdk-s3 CONFIG REQUIRED)
find_package(aws-cpp-sdk-core CONFIG REQUIRED)

# SimpleAmqpClient installs only a pkg-config file (no CMake config).
find_package(PkgConfig REQUIRED)
pkg_check_modules(SAC REQUIRED IMPORTED_TARGET libSimpleAmqpClient)

# ── System libcurl workaround ─────────────────────────────────────────────
# Ubuntu's system OpenCV may pull GDAL/NetCDF built against system libcurl4 versioned symbols
# (CURL_OPENSSL_4). vcpkg libcurl lacks those versioned symbols → unresolved symbols at link.
# Fix: prefer system libcurl when linking OpenCV targets on Linux.
set(FALL_CURL_FOR_OPENCV CURL::libcurl)
if(UNIX AND NOT APPLE)
    find_library(FALL_SYSTEM_CURL NAMES curl
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib64 /lib
        NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_INSTALL_PREFIX NO_CMAKE_SYSTEM_PATH)
    if(NOT FALL_SYSTEM_CURL OR FALL_SYSTEM_CURL MATCHES "vcpkg_installed")
        unset(FALL_SYSTEM_CURL CACHE)
        foreach(_c IN ITEMS
                /usr/lib/x86_64-linux-gnu/libcurl.so.4
                /lib/x86_64-linux-gnu/libcurl.so.4
                /usr/lib64/libcurl.so.4)
            if(EXISTS "${_c}")
                set(FALL_SYSTEM_CURL "${_c}" CACHE FILEPATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(FALL_SYSTEM_CURL AND NOT FALL_SYSTEM_CURL MATCHES "vcpkg_installed")
        add_library(fall_system_curl SHARED IMPORTED)
        set_target_properties(fall_system_curl PROPERTIES IMPORTED_LOCATION "${FALL_SYSTEM_CURL}")
        set(FALL_CURL_FOR_OPENCV fall_system_curl)
        message(STATUS "OpenCV: using system libcurl (${FALL_SYSTEM_CURL}) for GDAL/NetCDF symbol versions")
    endif()
endif()

# ── Source lists ────────────────────────────────────────────────
set(COMMON_SRC
    app/config.cpp
    app/utils/logger.cpp
    app/core/orchestrators/zmq_io.cpp
)

set(FALL_SRC
    ${COMMON_SRC}
    app/core/helpers.cpp
    app/core/services/fall_eval.cpp
    app/core/services/frame_buffer_manager.cpp
    app/core/services/fall.cpp
    app/core/inferences/pose.cpp
)

# ── Executables ─────────────────────────────────────────────────
add_executable(fall_inference
    app/core/orchestrators/fall.cpp
    ${FALL_SRC}
)
target_include_directories(fall_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fall_inference PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    libzmq
    nlohmann_json::nlohmann_json
    ${FALL_CURL_FOR_OPENCV}
)

add_executable(camera_reader
    app/core/orchestrators/camera.cpp
    ${COMMON_SRC}
    app/utils/rtsp_camera.cpp
)
target_include_directories(camera_reader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(camera_reader PRIVATE
    ${OpenCV_LIBS}
    libzmq
    nlohmann_json::nlohmann_json
    redis++::redis++
    ${FALL_CURL_FOR_OPENCV}
)

add_executable(msg_gen
    app/core/orchestrators/msg_gen.cpp
    ${COMMON_SRC}
    app/core/services/video_helper.cpp
    app/utils/aws.cpp
    app/kafka/msk_token.cpp
    app/kafka/kafka_producer.cpp
)
target_include_directories(msg_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(msg_gen PRIVATE
    ${OpenCV_LIBS}
    libzmq
    nlohmann_json::nlohmann_json
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    RdKafka::rdkafka++
    PkgConfig::SAC
    ${FALL_CURL_FOR_OPENCV}
)

option(BUILD_ANTMEDIA_EXECUTABLE "Build antmedia_service binary" OFF)
if(BUILD_ANTMEDIA_EXECUTABLE)
    add_executable(antmedia_service
        antmedia_service/main.cpp
        antmedia_service/TokenManager.cpp
        antmedia_service/Pipeline.cpp
    )
    target_include_directories(antmedia_service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(antmedia_service PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        ${FALL_CURL_FOR_OPENCV}
    )
endif()
