from pathlib import Path

import numpy as np
import open3d as o3d
from pypcd4 import PointCloud

from oxford_spires_utils.point_cloud import transform_3d_cloud
from oxford_spires_utils.se3 import se3_matrix_to_xyz_quat_wxyz, xyz_quat_wxyz_to_se3_matrix


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


if __name__ == "__main__":
    pcd_folder_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/input_cloud_test"
    new_pcd_folder_path = Path(pcd_folder_path).parent / (Path(pcd_folder_path).name + "_new")

    # transform_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    se3_matrix_txt_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/T_rtc_lidar"
    transform_matrix = np.loadtxt(se3_matrix_txt_path)

    transform_pcd_folder(pcd_folder_path, new_pcd_folder_path, transform_matrix)
