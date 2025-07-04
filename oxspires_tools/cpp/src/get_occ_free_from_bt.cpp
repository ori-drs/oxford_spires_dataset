#include "oxspires_tools/get_occ_free_from_bt.h"

using namespace std;
using namespace octomap;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

void convertOctreeToPointCloud(const std::string bt_file_path, const std::string free_pcd_path,
                               const std::string occupied_pcd_path) {
  // Load the .bt file into an OcTree
  OcTree tree(bt_file_path);

  if (!tree.size()) {
    cerr << "Failed to load the .bt file or the tree is empty." << endl;
    return;
  }

  // Retrieve and print the thresholds
  double occupancyThreshold = tree.getOccupancyThres();
  double freeThreshold = tree.getProbMiss(); // ProbMiss is related to how free space is handled

  cout << "Occupancy threshold: " << occupancyThreshold << endl;  // Default is 0.5
  cout << "Free threshold (probMiss): " << freeThreshold << endl; // Default is 0.12

  // Point clouds for free and occupied voxels
  PointCloud::Ptr free_cloud(new PointCloud);
  PointCloud::Ptr occupied_cloud(new PointCloud);

  // Get the total number of leaf nodes for the progress bar
  size_t total_leafs = tree.getNumLeafNodes();
  size_t processed_leafs = 0;

  // Iterate through all leaf nodes
  for (OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
    point3d coord = it.getCoordinate();

    // Classify voxel as free, occupied, or unknown and add to corresponding cloud
    if (tree.isNodeOccupied(*it)) {
      occupied_cloud->points.emplace_back(coord.x(), coord.y(), coord.z());
    } else {
      free_cloud->points.emplace_back(coord.x(), coord.y(), coord.z());
    }

    // Update the progress bar
    processed_leafs++;
    if (processed_leafs % 100 == 0 || processed_leafs == total_leafs) { // Update every 100 iterations
      displayProgressBar(processed_leafs, total_leafs);
    }
  }

  // Final progress bar completion
  displayProgressBar(total_leafs, total_leafs);
  cout << endl;

  cout << "Size of free_cloud: " << free_cloud->points.size() << endl;
  cout << "Size of occupied_cloud: " << occupied_cloud->points.size() << endl;

  // Set cloud properties
  free_cloud->width = free_cloud->points.size();
  free_cloud->height = 1;
  free_cloud->is_dense = true;

  occupied_cloud->width = occupied_cloud->points.size();
  occupied_cloud->height = 1;
  occupied_cloud->is_dense = true;

  // Save the point clouds as PCD files with specified file paths
  cout << "Saving free cloud with " << free_cloud->points.size() << " points to " << free_pcd_path << "..." << endl;
  pcl::io::savePCDFileASCII(free_pcd_path, *free_cloud);

  cout << "Saving occupied cloud with " << occupied_cloud->points.size() << " points to " << occupied_pcd_path << "..."
       << endl;
  pcl::io::savePCDFileASCII(occupied_pcd_path, *occupied_cloud);

  cout << "\nPoint clouds saved as:" << endl;
  cout << "Free voxels: " << free_pcd_path << " (" << free_cloud->points.size() << " points)" << endl;
  cout << "Occupied voxels: " << occupied_pcd_path << " (" << occupied_cloud->points.size() << " points)" << endl;
}
