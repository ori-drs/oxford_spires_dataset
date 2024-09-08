#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h> 
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <Eigen/Geometry>

// Function to list files in a directory
std::vector<std::string> listFiles(const std::string& folderPath) {
    std::vector<std::string> files;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(folderPath.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".pcd") {
                files.push_back(folderPath + "/" + filename);
            }
        }
        closedir(dir);
    }
    return files;
}

void processPCDFolder(const std::string& folderPath, double resolution = 0.1) {
    octomap::OcTree tree(resolution);

    std::vector<std::string> pcdFiles = listFiles(folderPath);

    // Sort the PCD files
    std::sort(pcdFiles.begin(), pcdFiles.end());

    for (const auto& filePath : pcdFiles) {
        std::string filename = filePath.substr(filePath.find_last_of("/") + 1);
        std::cout << "Processing file: " << filename << std::endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filePath, *cloud) == -1) {
            std::cerr << "Couldn't read file " << filename << std::endl;
            continue;
        }

        std::cout << "Point cloud size: " << cloud->size() << std::endl;

        // Extract viewpoint directly from the PCD file
        Eigen::Vector4f origin = cloud->sensor_origin_;
        Eigen::Quaternionf orientation = cloud->sensor_orientation_;

        std::cout << "Viewpoint: " 
                  << origin.x() << ", " 
                  << origin.y() << ", " 
                  << origin.z() << ", "
                  << orientation.x() << ", "
                  << orientation.y() << ", "
                  << orientation.z() << ", "
                  << orientation.w() << std::endl;

        // Convert origin to octomap point3d
        octomap::point3d sensor_origin(origin.x(), origin.y(), origin.z());

        // Transform the point cloud using the origin and orientation
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() << origin.x(), origin.y(), origin.z();
        transform.rotate(orientation);

        // Apply the transformation to the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

        // Convert transformed PCL cloud to OctoMap point cloud
        octomap::Pointcloud octomap_cloud;
        for (const auto& point : transformed_cloud->points) {
            octomap_cloud.push_back(point.x, point.y, point.z);
        }

        // Insert the transformed cloud into the OctoMap
        tree.insertPointCloud(octomap_cloud, sensor_origin, -1, true, true);
    }

    // Optionally, you can save the resulting octomap
    tree.writeBinary("result_octomap.bt");
    std::cout << "Octomap saved to result_octomap.bt" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_pcd_folder> <pcd/e75> [resolution]" << std::endl;
        return 1;
    }

    double resolution = 0.1;
    if (argc > 2) {
        resolution = std::stod(argv[2]);
    }

    processPCDFolder(argv[1], resolution);

    return 0;
}