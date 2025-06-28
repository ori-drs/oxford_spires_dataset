#ifndef PROCESS_PCD_FOLDER_H
#define PROCESS_PCD_FOLDER_H

#include <Eigen/Geometry>
#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

// Function to process PCD files in a folder and generate an OctoMap
void processPCDFolder(const std::string &folderPath, double resolution, const std::string &save_path);

#endif // PROCESS_PCD_FOLDER_H
