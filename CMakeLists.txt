cmake_minimum_required(VERSION 3.15...3.27)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED
  COMPONENTS
  common
  filters
  io
  kdtree
  sample_consensus
  segmentation
  surface
  search
)

include_directories(
    "oxford_spires_utils/cpp"
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

# Add modules
pybind11_add_module(cpp
    "oxford_spires_utils/cpp/main.cpp"
)
target_compile_features(cpp PUBLIC cxx_std_14)
target_link_libraries(cpp PRIVATE pybind11::headers ${PCL_LIBRARIES})

# Install
install(TARGETS cpp LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})