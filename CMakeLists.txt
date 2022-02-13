cmake_minimum_required(VERSION 2.8)
project(RGBDDirectMethod)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 ${SSE_FLAGS} -g -O3")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # Add elements to list
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED) # used to show pointcloud
# find_package(CSparse REQUIRED)
# include_directories(${CSPARSE_INCLUDE_DIR})
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS}) 

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse Sophus::Sophus)
