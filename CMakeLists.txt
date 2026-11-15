cmake_minimum_required(VERSION 3.16)
project(ppocr6-studio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_GUI "Build Qt5 parameter tuning GUI (ppocr6-tuner)" OFF)
option(FETCH_ONNXRUNTIME "Automatically download ONNX Runtime if not found locally" ON)
set(ONNXRUNTIME_VERSION "1.32.0" CACHE STRING "ONNX Runtime version to fetch if not found")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional path to prebuilt ONNX Runtime root directory")

# Keep runtime libraries alongside executables on Unix
if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
elseif(APPLE)
    set(CMAKE_BUILD_RPATH "@executable_path")
endif()

# ------- OpenCV -------
# Works with apt (libopencv-dev), vcpkg (opencv4), or -DOpenCV_DIR=...
find_package(OpenCV REQUIRED)

# ------- ONNX Runtime -------
set(_ORT_LIB_DIR "")

if(NOT ONNXRUNTIME_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
endif()

if(ONNXRUNTIME_ROOT)
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")
    set(_ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(_ORT_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
else()
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "Found ONNX Runtime via find_package")
    elseif(FETCH_ONNXRUNTIME)
        include(FetchContent)
        if(WIN32)
            set(_ORT_PLATFORM "win-x64")
            set(_ORT_EXT "zip")
        elseif(APPLE)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                set(_ORT_PLATFORM "osx-arm64")
            else()
                set(_ORT_PLATFORM "osx-x86_64")
            endif()
            set(_ORT_EXT "tgz")
        else()
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(_ORT_PLATFORM "linux-aarch64")
            else()
                set(_ORT_PLATFORM "linux-x64")
            endif()
            set(_ORT_EXT "tgz")
        endif()

        set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${_ORT_PLATFORM}-${ONNXRUNTIME_VERSION}.${_ORT_EXT}")
        message(STATUS "ONNX Runtime not found locally; fetching ${_ORT_URL}")

        FetchContent_Declare(
            onnxruntime_prebuilt
            URL "${_ORT_URL}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(onnxruntime_prebuilt)

        set(_ORT_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/include")
        set(_ORT_LIB_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/lib")
    else()
        message(FATAL_ERROR
            "ONNX Runtime not found. Install via vcpkg, pass -DONNXRUNTIME_ROOT=<path>, "
            "or enable -DFETCH_ONNXRUNTIME=ON.")
    endif()
endif()

if(TARGET onnxruntime::onnxruntime)
    set(_ORT_TARGET onnxruntime::onnxruntime)
elseif(TARGET onnxruntime)
    set(_ORT_TARGET onnxruntime)
else()
    add_library(onnxruntime SHARED IMPORTED)
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ORT_LIB_DIR}/onnxruntime.dll"
            IMPORTED_IMPLIB "${_ORT_LIB_DIR}/onnxruntime.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${_ORT_INCLUDE_DIR}"
        )
    elseif(APPLE)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ORT_LIB_DIR}/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${_ORT_INCLUDE_DIR}"
        )
    else()
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ORT_LIB_DIR}/libonnxruntime.so"
            INTERFACE_INCLUDE_DIRECTORIES "${_ORT_INCLUDE_DIR}"
        )
    endif()
    set(_ORT_TARGET onnxruntime)
endif()

# Helper to copy ONNX Runtime shared libraries next to a target
function(ppocr_copy_ort_runtime target_name)
    if(NOT _ORT_LIB_DIR)
        return()
    endif()
    if(WIN32)
        file(GLOB _ORT_DLLS "${_ORT_LIB_DIR}/onnxruntime*.dll")
        foreach(_dll ${_ORT_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()
    else()
        file(GLOB _ORT_SOS "${_ORT_LIB_DIR}/libonnxruntime*.so*" "${_ORT_LIB_DIR}/libonnxruntime*.dylib")
        foreach(_so ${_ORT_SOS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_so}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()
    endif()
endfunction()

# ------- Core Library (ppocr6::core) -------
add_library(ocr_core STATIC
    src/core/ocr_engine.cpp
    src/core/ocr_engine.h
    src/core/params.h
)
add_library(ppocr6::core ALIAS ocr_core)
set_target_properties(ocr_core PROPERTIES OUTPUT_NAME ppocr6_core)
target_include_directories(ocr_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
    $<INSTALL_INTERFACE:include/ppocr6>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ocr_core PUBLIC
    ${_ORT_TARGET}
    ${OpenCV_LIBS}
)

# Optional install rules for third-party integration
install(TARGETS ocr_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES src/core/ocr_engine.h src/core/params.h
    DESTINATION include/ppocr6
)

# ------- CLI Executable (ppocr6) -------
add_executable(ppocr6 src/cli/main.cpp)
target_link_libraries(ppocr6 PRIVATE ocr_core)
ppocr_copy_ort_runtime(ppocr6)

# ------- Optional Qt5 GUI Executable (ppocr6-tuner) -------
if(BUILD_GUI)
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
    set(CMAKE_AUTOMOC ON)
    add_executable(ppocr6-tuner
        src/gui/main.cpp
        src/gui/mainwindow.cpp
        src/gui/mainwindow.h
    )
    target_link_libraries(ppocr6-tuner PRIVATE
        ocr_core
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
    )
    ppocr_copy_ort_runtime(ppocr6-tuner)
endif()
