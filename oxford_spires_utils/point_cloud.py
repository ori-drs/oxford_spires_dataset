from pathlib import Path

import open3d as o3d
from tqdm import tqdm

from oxford_spires_utils.io import read_pcd_with_viewpoint


def merge_downsample_vilens_slam_clouds(vilens_slam_clouds_folder, output_cloud_path=None, downsample_voxel_size=0.05):
    cloud_paths = list(Path(vilens_slam_clouds_folder).rglob("*.pcd"))
    print("Merging clouds ...")
    final_cloud = o3d.geometry.PointCloud()
    for cloud_path in tqdm(cloud_paths):
        cloud = read_pcd_with_viewpoint(str(cloud_path))
        final_cloud += cloud
    print(f"Downsampling to {downsample_voxel_size}m ...")

    if output_cloud_path is None:
        output_cloud_path = Path(vilens_slam_clouds_folder).parent / f"merged_{downsample_voxel_size}m.ply"
    final_cloud = final_cloud.voxel_down_sample(voxel_size=downsample_voxel_size)
    print(f"Saving merged cloud to {output_cloud_path} ...")
    o3d.io.write_point_cloud(str(output_cloud_path), final_cloud)
