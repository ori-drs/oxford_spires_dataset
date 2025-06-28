#include <chrono>
#include <fstream>
#include <iostream>
#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "progress_bar.h"

void convertOctreeToPointCloud(const std::string bt_file_path, const std::string free_pcd_path,
                               const std::string occupied_pcd_path);
