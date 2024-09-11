cmake_minimum_required(VERSION 3.10)

project(spires_octomap_utils)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find OctoMap package
find_package(Octomap REQUIRED)
find_package(PCL REQUIRED)


# Include the OctoMap headers
include_directories(${OCTOMAP_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})


add_executable(pcd2bt pcd2bt.cpp)
target_link_libraries(pcd2bt ${OCTOMAP_LIBRARIES} ${PCL_LIBRARIES})

add_executable(get_occ_free_from_bt get_occ_free_from_bt.cpp)
target_link_libraries(get_occ_free_from_bt ${OCTOMAP_LIBRARIES} ${PCL_LIBRARIES})