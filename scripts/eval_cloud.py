from pathlib import Path

import numpy as np
import open3d as o3d

from oxspires_tools.eval import save_error_cloud


def evaluate_cloud(input_cloud_path, gt_cloud_path):
    gt_cloud = o3d.io.read_point_cloud(str(gt_cloud_path))
    input_cloud = o3d.io.read_point_cloud(str(input_cloud_path))
    save_error_cloud_path = str(Path(input_cloud_path).with_name(f"{Path(input_cloud_path).stem}_error.ply"))
    input_cloud_np = np.array(input_cloud.points)
    gt_cloud_np = np.array(gt_cloud.points)
    save_error_cloud(input_cloud_np, gt_cloud_np, save_error_cloud_path)


if __name__ == "__main__":
    gt_cloud_path = "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/rtc_gt_colmap_frame.ply"
    input_cloud_path = "/home/yifu/workspace/T-RO_2025/2024-03-18-chch-4/nerf_rgb_merged.ply"

    evaluate_cloud(input_cloud_path, gt_cloud_path)
