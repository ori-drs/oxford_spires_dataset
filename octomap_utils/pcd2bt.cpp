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

void processPCDFolder(const std::string& folderPath, double resolution, const std::string& save_path, const Eigen::Affine3f& custom_transform) {
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
        pcl::transformPointCloud(*transformed_cloud, *transformed_cloud, custom_transform);

        // Convert transformed PCL cloud to OctoMap point cloud
        octomap::Pointcloud octomap_cloud;
        for (const auto& point : transformed_cloud->points) {
            octomap_cloud.push_back(point.x, point.y, point.z);
        }

        // Insert the transformed cloud into the OctoMap
        tree.insertPointCloud(octomap_cloud, sensor_origin, -1, true, true);
    }

    // Save the resulting octomap
    tree.writeBinary(save_path);
    std::cout << std::endl;
    std::cout << "Octomap saved to " << save_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_pcd_folder> -r [resolution] -s [saved_path] -tf [transform_path]" << std::endl;
        return 1;
    }
    std::string folderPath = argv[1];
    double resolution = 0.1;
    std::string save_path = "result_octomap.bt";
    Eigen::Affine3f custom_transform = Eigen::Affine3f::Identity();
    bool transform_set = false;

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r" && i + 1 < argc) {
            resolution = std::stod(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (arg == "-tf" && i + 7 < argc) {
            float tx = std::stof(argv[++i]);
            float ty = std::stof(argv[++i]);
            float tz = std::stof(argv[++i]);
            float qw = std::stof(argv[++i]);
            float qx = std::stof(argv[++i]);
            float qy = std::stof(argv[++i]);
            float qz = std::stof(argv[++i]);

            custom_transform = Eigen::Affine3f::Identity();
            custom_transform.translation() << tx, ty, tz;
            Eigen::Quaternionf quaternion(qw, qx, qy, qz);
            custom_transform.rotate(quaternion);

            std::cout << "Applied custom transform: "
                      << "t(" << tx << ", " << ty << ", " << tz << ") "
                      << "quat(" << qw << ", " << qx << ", " << qy << ", " << qz << ")" << std::endl;

            transform_set = true;
        } else {
            std::cerr << "Unknown argument or missing value: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " <path_to_pcd_folder> -r [resolution] -s [saved_path] -tf x y z quat_wxyz" << std::endl;
            return 1;
        }
    }

    // If no transformation is provided, use the identity matrix
    if (!transform_set) {
        std::cout << "No custom transform provided. Using identity transformation." << std::endl;
    }
    std::cout << " transformation:\n" << custom_transform.matrix() << std::endl;

    processPCDFolder(folderPath, resolution, save_path, custom_transform);

    return 0;
}
