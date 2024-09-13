#include <iostream>
#include <octomap/OcTree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void removeUnknownPoints(pcl::PointCloud<pcl::PointXYZ> &cloud, const octomap::OcTree &tree,
                         const std::string &output_file);
