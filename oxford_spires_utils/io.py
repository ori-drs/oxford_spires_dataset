import numpy as np
import open3d as o3d

from oxford_spires_utils.se3 import xyz_quat_xyzw_to_se3_matrix

np.set_printoptions(suppress=True, precision=4)


def read_pcd_with_viewpoint(file_path: str):
    assert file_path.endswith(".pcd")
    with open(file_path, mode="rb") as file:
        while True:
            line = file.readline().decode("utf-8").strip()
            if line.startswith("VIEWPOINT"):
                viewpoint = line.split()[1:]  # x y z qw qx qy qz
                xyz = viewpoint[:3]
                quat_wxyz = viewpoint[3:]
                quat_xyzw = quat_wxyz[1:] + [quat_wxyz[0]]
                se3_matrix = xyz_quat_xyzw_to_se3_matrix(xyz, quat_xyzw)
                break
    cloud = o3d.io.read_point_cloud(file_path)
    cloud.transform(se3_matrix)
    return cloud
