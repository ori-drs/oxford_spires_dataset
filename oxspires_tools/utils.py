from pathlib import Path

import numpy as np
import open3d as o3d
from pypcd4 import PointCloud
from tqdm import tqdm

from oxspires_tools.point_cloud import convert_e57_to_pcd, transform_3d_cloud
from oxspires_tools.se3 import se3_matrix_to_xyz_quat_wxyz, xyz_quat_wxyz_to_se3_matrix
from oxspires_tools.trajectory.pose_convention import PoseConvention


def convert_e57_folder_to_pcd_folder(e57_folder, pcd_folder):
    Path(pcd_folder).mkdir(parents=True, exist_ok=True)
    e57_files = list(Path(e57_folder).glob("*.e57"))
    pbar = tqdm(e57_files)
    print(f"Converting {len(e57_files)} E57 files in {e57_folder} to PCD files in {pcd_folder}")
    for e57_file in pbar:
        pcd_file = Path(pcd_folder) / (e57_file.stem + ".pcd")
        pbar.set_description(f"Processing {e57_file.name}")
        convert_e57_to_pcd(e57_file, pcd_file, pcd_lib="pypcd4")


def transform_pcd_folder(folder_path, new_folder_path, transform_matrix):
    """Process all PCD files in the given folder with the predefined transform."""
    pcd_files = sorted(list(Path(folder_path).rglob("*.pcd")))
    new_folder_path.mkdir(exist_ok=True)
    for filename in pcd_files:
        pc = PointCloud.from_path(filename)
        viewpoint = list(pc.metadata.viewpoint)
        viewpoint_se3 = xyz_quat_wxyz_to_se3_matrix(viewpoint[:3], viewpoint[3:])
        new_viewpoint_se3 = transform_matrix @ viewpoint_se3
        new_viewpoint = se3_matrix_to_xyz_quat_wxyz(new_viewpoint_se3)
        new_viewpoint = np.concatenate((new_viewpoint[0], new_viewpoint[1]))
        new_viewpoint = new_viewpoint.tolist()

        points = pc.numpy(("x", "y", "z"))
        transformed_points = transform_3d_cloud(points, transform_matrix)

        pc = o3d.io.read_point_cloud(str(filename))
        assert np.allclose(np.asarray(pc.points), points)
        assert np.allclose(np.asarray(pc.transform(transform_matrix).points), transformed_points)
        # colour = np.asarray(pc.colors)
        # colour_encoded = PointCloud.encode_rgb(colour)
        # colour_encoded = colour_encoded.reshape(-1, 1)
        # points = np.hstack((points, colour_encoded))

        saved_pcd4 = PointCloud.from_xyz_points(points)
        saved_pcd4.metadata.viewpoint = new_viewpoint

        new_filename = new_folder_path / filename.name
        saved_pcd4.save(new_filename)


def get_nerf_pose(colmap_c2w):
    # equivalent to https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py
    # new local frame
    vision_2_graphics = PoseConvention.transforms["vision"]["graphics"]
    # new global frame # TODO this is unnecessary
    world_transform = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return world_transform @ colmap_c2w @ vision_2_graphics
