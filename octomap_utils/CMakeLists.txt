cmake_minimum_required(VERSION 3.10)

project(spires_octomap_utils)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find OctoMap package
find_package(Octomap REQUIRED)
find_package(PCL REQUIRED)


include_directories(
    include
    ${OCTOMAP_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  progress_bar.cpp
)
target_link_libraries(${PROJECT_NAME} ${OCTOMAP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

add_executable(pcd2bt pcd2bt.cpp)
target_link_libraries(pcd2bt ${PROJECT_NAME})

add_executable(get_occ_free_from_bt get_occ_free_from_bt.cpp)
target_link_libraries(get_occ_free_from_bt ${PROJECT_NAME})

add_executable(filter_pcd_with_bt filter_pcd_with_bt.cpp )
target_link_libraries(filter_pcd_with_bt ${PROJECT_NAME})