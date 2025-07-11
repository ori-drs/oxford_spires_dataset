cmake_minimum_required(VERSION 3.15...3.27)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

# if BUILD_CPP==0, skip and only install Python. Will not require pcl and octomap
if(NOT DEFINED ENV{BUILD_CPP} OR NOT $ENV{BUILD_CPP} STREQUAL "1")
message(STATUS "Skipping C++ build because BUILD_CPP=0")
return()
endif()

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

include_directories(
    "oxspires_tools/cpp/include"
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)

# Add modules
pybind11_add_module(cpp
    "oxspires_tools/cpp/main.cpp"
    "oxspires_tools/cpp/src/filter_pcd_with_bt.cpp"
    "oxspires_tools/cpp/src/get_occ_free_from_bt.cpp"
    "oxspires_tools/cpp/src/pcd2bt.cpp"
    "oxspires_tools/cpp/src/progress_bar.cpp"
)
target_compile_features(cpp PUBLIC cxx_std_14)
target_link_libraries(cpp PRIVATE pybind11::headers ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

# Install
install(TARGETS cpp LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})