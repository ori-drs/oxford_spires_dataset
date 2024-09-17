from __future__ import annotations

from spires_cpp import (
    OcTree,
    convertOctreeToPointCloud,
    processPCDFolder,
    removeUnknownPoints,
)

pcd_folder = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw/individual_clouds_new"
resolution = 0.1
save_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw/individual_clouds_new.bt"
processPCDFolder(pcd_folder, resolution, save_path)


octomap_bt_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw/individual_clouds_new.bt"
octree = OcTree(octomap_bt_path)
octree.getResolution()
octree.size()
octree.getTreeDepth()


octomap_bt_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/gt_cloud.bt"
input_cloud_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/input_cloud_occ.pcd"
output_cloud_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/input_cloud_occ_filtered.pcd"
removeUnknownPoints(input_cloud_path, octomap_bt_path, output_cloud_path)


octomap_bt_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/gt_cloud.bt"
occ_cloud_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/gt_cloud_occ.pcd"
free_cloud_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/gt_cloud_free.pcd"
convertOctreeToPointCloud(octomap_bt_path, free_cloud_path, occ_cloud_path)
