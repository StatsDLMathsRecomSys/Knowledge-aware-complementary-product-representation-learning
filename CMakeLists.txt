cmake_minimum_required(VERSION 3.0)

project(Univector)

set(CMAKE_CXX_FLAGS "-Wall --std=c++11 -Wno-reorder -pthread -funroll-loops -O3 -march=native -ffast-math")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)


find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})


file(GLOB CNPY_SOURCE
"third_party/cnpy/*.cpp"
"third_party/cnpy/*.h"
)
add_library(cnpy STATIC ${CNPY_SOURCE})
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
target_include_directories(cnpy PUBLIC  ${PROJECT_SOURCE_DIR}/third_party)


file(GLOB UNI_SOURCE
"src/*.cpp"
"src/*.h"
)
add_library(univec STATIC ${UNI_SOURCE})
target_include_directories(univec PUBLIC  ${PROJECT_SOURCE_DIR}/src)
target_include_directories(univec PUBLIC  ${EIGEN3_INCLUDE_DIR})

target_link_libraries(univec cnpy)



add_executable(uni-vec app/main.cpp)
target_link_libraries(uni-vec univec)