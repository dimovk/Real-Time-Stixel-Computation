cmake_minimum_required(VERSION 3.5.1)
project(stixels-live)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

SET(EXECUTABLE_OUTPUT_PATH ".")

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

SET(VERSION_REQ_CUDA "9.0")

SET(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_35,code=sm_35
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_70,code=sm_70
)

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package( CUDA 9.0 REQUIRED )

include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -msse4.1")
endif (CMAKE_COMPILER_IS_GNUCXX)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB srcs *.cpp *.hpp *.h*)

add_executable(stixels-live ${srcs})
add_definitions(-std=c++14 -O3)

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

target_link_libraries(stixels-live 
                        ${ZED_LIBS} 
                        ${OpenCV_LIBS} 
                        ${CUDA_LIBRARIES})
