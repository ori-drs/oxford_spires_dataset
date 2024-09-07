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


def modify_pcd_viewpoint(file_path: str, new_file_path: str, new_viewpoint_xyz_wxyz: np.ndarray):
    """
    open3d does not support writing viewpoint to pcd file, so we need to modify the pcd file manually.
    """
    assert file_path.endswith(".pcd")
    assert new_viewpoint_xyz_wxyz.shape == (7,), f"{new_viewpoint_xyz_wxyz} should be t_xyz_quat_wxyz"
    new_viewpoint = new_viewpoint_xyz_wxyz
    header_lines = []
    with open(file_path, mode="rb") as file:
        line = file.readline().decode("utf-8").strip()
        while line != "DATA binary":
            header_lines.append(line)
            line = file.readline().decode("utf-8").strip()
        binary_data = file.read()
    for i in range(len(header_lines)):
        if header_lines[i].startswith("VIEWPOINT"):
            header_lines[i] = (
                f"VIEWPOINT {new_viewpoint[0]} {new_viewpoint[1]} {new_viewpoint[2]} {new_viewpoint[3]} {new_viewpoint[4]} {new_viewpoint[5]} {new_viewpoint[6]}"
            )
            break
    with open(new_file_path, mode="wb") as file:
        for line in header_lines:
            file.write(f"{line}\n".encode("utf-8"))
        file.write(b"DATA binary\n")
        file.write(binary_data)
