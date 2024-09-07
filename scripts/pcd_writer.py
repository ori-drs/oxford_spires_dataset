import numpy as np
import open3d as o3d
import pye57
from scipy.spatial.transform import Rotation

from oxford_spires_utils.io import modify_pcd_viewpoint
from oxford_spires_utils.se3 import xyz_quat_xyzw_to_se3_matrix


def convert_e57_to_pcd(e57_file_path, pcd_file_path):
    # Load E57 file
    e57_file = pye57.E57(e57_file_path)

    # Get the first point cloud (assuming the E57 file contains at least one)
    data = e57_file.read_scan(0, intensity=True, colors=True)

    header = e57_file.get_header(0)
    t_xyz = header.translation
    quat_wxyz = header.rotation
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    assert np.allclose(rot_matrix, header.rotation_matrix)

    viewpoint_matrix = xyz_quat_xyzw_to_se3_matrix(t_xyz, quat_xyzw)
    viewpoint = np.concatenate((t_xyz, quat_wxyz))

    # Extract Cartesian coordinates
    x = data["cartesianX"]
    y = data["cartesianY"]
    z = data["cartesianZ"]

    # Create a numpy array of points
    points_np = np.vstack((x, y, z)).T
    points_homogeneous = np.hstack((points_np, np.ones((points_np.shape[0], 1))))
    points_sensor_frame = (np.linalg.inv(viewpoint_matrix) @ points_homogeneous.T).T[:, :3]

    colours = np.vstack((data["colorRed"], data["colorGreen"], data["colorBlue"])).T

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_sensor_frame)
    # pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colours / 255)

    # Save the point cloud as a PCD file
    o3d.io.write_point_cloud(pcd_file_path, pcd)
    print(f"PCD file saved to {pcd_file_path}")
    modify_pcd_viewpoint(pcd_file_path, pcd_file_path, viewpoint)


if __name__ == "__main__":
    e57_file_path = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual/Math Inst- 001.e57"
    output_pcd = "/home/yifu/workspace/oxford_spires_dataset/output.pcd"
    new_pcd = "/home/yifu/workspace/oxford_spires_dataset/output_new.pcd"
    convert_e57_to_pcd(e57_file_path, output_pcd)
