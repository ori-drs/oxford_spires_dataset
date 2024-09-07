import numpy as np
import open3d as o3d
import pye57
from scipy.spatial.transform import Rotation

from oxford_spires_utils.se3 import xyz_quat_xyzw_to_se3_matrix


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


def convert_e57_to_pcd(e57_file_path, pcd_file_path, check_output=True):
    # Load E57 file
    e57_file_path, pcd_file_path = str(e57_file_path), str(pcd_file_path)
    e57_file = pye57.E57(e57_file_path)

    header = e57_file.get_header(0)
    t_xyz = header.translation
    quat_wxyz = header.rotation
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    assert np.allclose(rot_matrix, header.rotation_matrix)

    viewpoint_matrix = xyz_quat_xyzw_to_se3_matrix(t_xyz, quat_xyzw)
    viewpoint = np.concatenate((t_xyz, quat_wxyz))

    has_colour = "colorRed" in header.point_fields
    # Get the first point cloud (assuming the E57 file contains at least one)
    data = e57_file.read_scan(0, intensity=False, colors=has_colour)

    # Extract Cartesian coordinates
    x = data["cartesianX"]
    y = data["cartesianY"]
    z = data["cartesianZ"]

    # Create a numpy array of points
    points_np = np.vstack((x, y, z)).T
    points_homogeneous = np.hstack((points_np, np.ones((points_np.shape[0], 1))))
    points_sensor_frame = (np.linalg.inv(viewpoint_matrix) @ points_homogeneous.T).T[:, :3]

    if has_colour:
        colours = np.vstack((data["colorRed"], data["colorGreen"], data["colorBlue"])).T

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sensor_frame)
    # pcd.points = o3d.utility.Vector3dVector(points_np)
    if has_colour:
        pcd.colors = o3d.utility.Vector3dVector(colours / 255)

    # Save the point cloud as a PCD file
    o3d.io.write_point_cloud(pcd_file_path, pcd)
    print(f"PCD file saved to {pcd_file_path}")
    modify_pcd_viewpoint(pcd_file_path, pcd_file_path, viewpoint)

    if check_output:
        saved_cloud = read_pcd_with_viewpoint(pcd_file_path)
        saved_cloud_np = np.array(saved_cloud.points)
        assert np.allclose(saved_cloud_np, points_np, rtol=1e-5, atol=1e-6)
        if has_colour:
            colours_np = np.array(saved_cloud.colors)
            assert np.allclose(colours_np, colours / 255, rtol=1e-5, atol=1e-8)
