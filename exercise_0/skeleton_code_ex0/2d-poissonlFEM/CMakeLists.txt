# Mandatory information that CMAKE needs
CMAKE_MINIMUM_REQUIRED(VERSION 3.12.0)


# Project name
PROJECT(2d-poissonlFEM)

include("cmake/GetEigen.cmake")
include("cmake/GetIGL.cmake")
include("cmake/warning_flags.cmake")
include("cmake/clang_tidy.cmake")

find_package(Threads REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Ignore specific warnings originating in Eigen or IGL
    add_definitions(-Wno-deprecated-register -Wno-return-type-c-linkage)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  # Ignore specific warnings originating in Eigen or IGL
    add_definitions(-Wno-deprecated-declarations -Wno-ignored-attributes)
endif()


add_executable(fem2d fem2d.cpp)
target_link_libraries(fem2d warning_flags)
target_link_libraries(fem2d ${THREADS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(fem2d Eigen IGL)
target_compile_definitions(fem2d PUBLIC -D_USE_MATH_DEFINES)
target_compile_definitions(fem2d PUBLIC -DANCSE_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")
target_compile_features(fem2d PUBLIC cxx_std_11)

if(CLANG_TIDY_EXE)
  set_target_properties(fem2d PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif()

add_subdirectory(unittest)
