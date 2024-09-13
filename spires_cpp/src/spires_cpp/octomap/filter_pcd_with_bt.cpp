#include "spires_cpp/octomap/filter_pcd_with_bt.h"

// Function to check if a point is unknown in the octree
bool isUnknownInOctree(const octomap::OcTree &tree, const octomap::point3d &point) {
  auto node = tree.search(point);
  // if the node does not exist, then it is unknown
  return !node;
}

// Function to remove unknown points from the point cloud
void removeUnknownPoints(pcl::PointCloud<pcl::PointXYZ> &cloud, const octomap::OcTree &tree,
                         const std::string &output_file) {

  pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

  for (const auto &pcl_point : cloud) {
    // Convert the PCL point to an octomap point
    octomap::point3d point(pcl_point.x, pcl_point.y, pcl_point.z);

    // Check if the point is unknown in the octree
    if (!isUnknownInOctree(tree, point)) {
      filtered_cloud.push_back(pcl_point);
    }
  }

  // Save the filtered point cloud to a file
  pcl::io::savePCDFileASCII(output_file, filtered_cloud);
  std::cout << "Filtered point cloud saved to " << output_file << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <octomap.bt> <input.pcd> <output.pcd>" << std::endl;
    return 1;
  }

  // Load the octree from the BT file
  octomap::OcTree tree(argv[1]);
  if (tree.size() == 0) {
    std::cerr << "Failed to load octomap from " << argv[1] << std::endl;
    return 1;
  }

  // Load the input point cloud
  pcl::PointCloud<pcl::PointXYZ> cloud;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], cloud) == -1) {
    std::cerr << "Failed to load point cloud from " << argv[2] << std::endl;
    return 1;
  }

  std::cout << "Loaded point cloud with " << cloud.size() << " points." << std::endl;

  // Remove unknown points from the point cloud
  removeUnknownPoints(cloud, tree, argv[3]);

  return 0;
}
