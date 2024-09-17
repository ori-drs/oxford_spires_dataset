#include "spires_cpp/octomap/filter_pcd_with_bt.h"

// Function to remove unknown points from the point cloud

void removeUnknownPoints(const std::string input_pcd_path, const std::string input_bt_path,
                         const std::string output_file_path) {
  octomap::OcTree tree(input_bt_path);
  if (tree.size() == 0) {
    std::cerr << "Failed to load octomap from " << input_bt_path << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ> cloud;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_pcd_path, cloud) == -1) {
    std::cout << "Couldn't read file " << input_pcd_path << std::endl;
    return;
  }
  pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

  for (const auto &pcl_point : cloud) {
    // Convert the PCL point to an octomap point
    octomap::point3d point(pcl_point.x, pcl_point.y, pcl_point.z);

    auto node = tree.search(point);
    bool is_unknown = !node;
    // Check if the point is unknown in the octree
    if (!is_unknown) {
      filtered_cloud.push_back(pcl_point);
    }
  }

  // Save the filtered point cloud to a file
  pcl::io::savePCDFileASCII(output_file_path, filtered_cloud);
  std::cout << "Filtered point cloud saved to " << output_file_path << std::endl;
}
