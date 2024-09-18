from pathlib import Path

import numpy as np

from oxford_spires_utils.eval import get_recon_metrics, save_error_cloud
from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds
from spires_cpp import convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints


def evaluate_lidar_cloud(
    project_folder, lidar_cloud_folder_path, gt_folder_path, octomap_resolution=0.1, downsample_voxel_size=0.05
):
    input_cloud_bt_path = Path(project_folder) / "input_cloud.bt"
    processPCDFolder(str(lidar_cloud_folder_path), octomap_resolution, str(input_cloud_bt_path))

    input_cloud_free_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_free.pcd"))
    input_cloud_occ_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ.pcd"))
    convertOctreeToPointCloud(str(input_cloud_bt_path), str(input_cloud_free_path), str(input_cloud_occ_path))

    input_cloud_merged_path = Path(project_folder) / "input_cloud_merged.ply"
    _ = merge_downsample_vilens_slam_clouds(lidar_cloud_folder_path, downsample_voxel_size, input_cloud_merged_path)
    input_cloud_filtered_path = Path(project_folder) / "input_cloud_merged_filtered.ply"
    removeUnknownPoints(str(input_cloud_merged_path), str(gt_cloud_bt_path), str(input_cloud_filtered_path))
    input_cloud_np = np.asarray(o3d.io.read_point_cloud(str(input_cloud_filtered_path)).points)
    gt_cloud_np = np.asarray(self.gt_cloud_merged.points)

    print(get_recon_metrics(input_cloud_np, gt_cloud_np))
    save_error_cloud(input_cloud_np, gt_cloud_np, str(Path(project_folder) / "input_error.ply"))
