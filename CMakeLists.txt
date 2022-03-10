cmake_minimum_required(VERSION 3.4...3.18)
project(DBow3)

set(OUTPUT_NAME py${PROJECT_NAME})


find_package(OpenMP)
if (OPENMP_FOUND)
        add_compile_options(-DUSE_OPENMP)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

SET(OpenCV_USE_STATIC_LIBS ON)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
link_libraries(
    ${OpenCV_LIBS}
)

set (CMAKE_CXX_STANDARD 11)

set(PYTHON_VERSION 3)
message( STATUS "PYTHON_VERSION = ${PYTHON_VERSION}")


# Add cmake_modules to module
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NumPy REQUIRED)


if (UNIX)
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  thirdparty/DBow3/src
 )
else()
  INCLUDE_DIRECTORIES(
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDES}
  thirdparty/DBow3/src
 )
endif()

add_subdirectory(thirdparty/DBow3/src)
add_subdirectory(thirdparty/DBow3/utils)

add_subdirectory(thirdparty/pybind11)

include_directories("thirdparty/pybind11/include")
include_directories("thirdparty/pybind11_opencv_numpy")

pybind11_add_module(${OUTPUT_NAME} "src/dbow3_wrapper.cpp" "./thirdparty/pybind11_opencv_numpy/ndarray_converter.cpp")
target_link_libraries(${OUTPUT_NAME} PRIVATE 
${PROJECT_NAME}
)
