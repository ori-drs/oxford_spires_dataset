from pathlib import Path
from tqdm import tqdm
import open3d as o3d
from oxford_spires_utils.io import read_pcd_with_viewpoint

if __name__ == "__main__":
    cloud_folder = "/home/yifu/data/nerf_data_pipeline/2024-03-18-chch_4/raw/individual_clouds"
    downsample_resolution = 0.05
    final_cloud_path = Path(cloud_folder).parent / f"merged_{downsample_resolution}m.ply"

    # cloud_paths = list(Path(cloud_folder).rglob("*.ply"))
    # cloud can be either ply or pcd

    pcd_cloud_paths = list(Path(cloud_folder).rglob("*.pcd"))
    ply_cloud_paths = list(Path(cloud_folder).rglob("*.ply"))
    cloud_paths = pcd_cloud_paths + ply_cloud_paths

    final_cloud = o3d.geometry.PointCloud()
    for cloud_path in tqdm(cloud_paths):
        cloud = read_pcd_with_viewpoint(str(cloud_path))
        final_cloud += cloud
    print(f"Downsampling to {downsample_resolution}m ...")
    final_cloud = final_cloud.voxel_down_sample(voxel_size=downsample_resolution)
    print(f"Saving merged cloud to {final_cloud_path} ...")
    o3d.io.write_point_cloud(str(final_cloud_path), final_cloud)
        
