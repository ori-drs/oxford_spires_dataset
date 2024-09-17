from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from oxford_spires_utils.io import read_pcd_with_viewpoint
from oxford_spires_utils.se3 import is_se3_matrix


def transform_3d_cloud(cloud_np, transform_matrix):
    """Apply a transformation to the point cloud."""
    # Convert points to homogeneous coordinates
    assert isinstance(cloud_np, np.ndarray)
    assert cloud_np.shape[1] == 3
    assert is_se3_matrix(transform_matrix)[0], is_se3_matrix(transform_matrix)[1]

    ones = np.ones((cloud_np.shape[0], 1))
    homogenous_points = np.hstack((cloud_np, ones))

    transformed_points = homogenous_points @ transform_matrix.T

    return transformed_points[:, :3]


def merge_downsample_clouds(cloud_path_list, output_cloud_path, downsample_voxel_size=0.05):
    print("Merging clouds ...")
    final_cloud = o3d.geometry.PointCloud()
    for cloud_path in tqdm(cloud_path_list):
        if cloud_path.endswith(".pcd"):
            cloud = read_pcd_with_viewpoint(str(cloud_path))
        elif cloud_path.endswith(".ply"):
            cloud = o3d.io.read_point_cloud(str(cloud_path))
        else:
            raise ValueError(f"Unsupported file format: {cloud_path}")
        final_cloud += cloud

    print(f"Downsampling to {downsample_voxel_size}m ...")
    final_cloud = final_cloud.voxel_down_sample(voxel_size=downsample_voxel_size)
    print(f"Saving merged cloud to {output_cloud_path} ...")
    o3d.io.write_point_cloud(str(output_cloud_path), final_cloud)
    return final_cloud


def merge_downsample_vilens_slam_clouds(vilens_slam_clouds_folder, downsample_voxel_size=0.05):
    cloud_paths = list(Path(vilens_slam_clouds_folder).rglob("*.pcd"))
    output_cloud_path = Path(vilens_slam_clouds_folder).parent / f"merged_{downsample_voxel_size}m.ply"
    return merge_downsample_clouds(cloud_paths, output_cloud_path, downsample_voxel_size)
