#include <iostream>
#include <octomap/OcTree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void removeUnknownPoints(const std::string input_pcd_path, const std::string input_bt_path,
                         const std::string output_file_path);