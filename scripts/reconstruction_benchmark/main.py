import multiprocessing
from pathlib import Path

import numpy as np
from utils import convert_e57_folder_to_pcd_folder

from oxford_spires_utils.eval import get_recon_metrics, save_error_cloud
from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds
from spires_cpp import convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints


def evaluate_lidar_cloud(project_folder, lidar_cloud_folder_path, gt_folder_path, octomap_resolution=0.1):
    Path(project_folder).mkdir(parents=True, exist_ok=True)
    input_cloud_bt_path = Path(project_folder) / "input_cloud.bt"
    gt_cloud_bt_path = Path(project_folder) / "gt_cloud.bt"

    processes = []
    for cloud_folder, cloud_bt_path in zip(
        [lidar_cloud_folder_path, gt_folder_path], [input_cloud_bt_path, gt_cloud_bt_path]
    ):
        process = multiprocessing.Process(
            target=processPCDFolder, args=(str(cloud_folder), octomap_resolution, str(cloud_bt_path))
        )
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    gt_cloud_free_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_free.pcd"))
    gt_cloud_occ_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_occ.pcd"))
    input_cloud_free_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_free.pcd"))
    input_cloud_occ_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ.pcd"))
    convertOctreeToPointCloud(str(gt_cloud_bt_path), str(gt_cloud_free_path), str(gt_cloud_occ_path))
    convertOctreeToPointCloud(str(input_cloud_bt_path), str(input_cloud_free_path), str(input_cloud_occ_path))

    input_occ_pcd_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ.pcd"))
    input_occ_filtered_path = str(
        Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ_filtered.pcd")
    )
    gt_occ_pcd_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_occ.pcd"))
    gt_occ_filtered_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_occ_filtered.pcd"))

    removeUnknownPoints(input_occ_pcd_path, str(gt_cloud_bt_path), input_occ_filtered_path)
    removeUnknownPoints(gt_occ_pcd_path, str(input_cloud_bt_path), gt_occ_filtered_path)

    downsample_voxel_size = 0.03
    input_cloud_np = np.asarray(merge_downsample_vilens_slam_clouds(lidar_cloud_folder_path, downsample_voxel_size))
    gt_cloud_np = np.asarray(merge_downsample_vilens_slam_clouds(gt_folder_path, downsample_voxel_size))
    print(get_recon_metrics(input_cloud_np, gt_cloud_np))
    save_error_cloud(input_cloud_np, gt_cloud_np, str(Path(project_folder) / "error_cloud_input.ply"))


if __name__ == "__main__":
    input_cloud_folder_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/individual_clouds"
    gt_cloud_folder_e57_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_e57"
    gt_cloud_folder_pcd_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_pcd"
    project_folder = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/recon_benchmark"
    convert_e57_folder_to_pcd_folder(gt_cloud_folder_e57_path, gt_cloud_folder_pcd_path)
    evaluate_lidar_cloud(project_folder, input_cloud_folder_path, gt_cloud_folder_pcd_path)
