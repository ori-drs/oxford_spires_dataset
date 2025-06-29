import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import pye57
from pypcd4 import PointCloud
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from oxspires_tools.se3 import is_se3_matrix, xyz_quat_xyzw_to_se3_matrix

logger = logging.getLogger(__name__)


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
        cloud_path = str(cloud_path)
        if cloud_path.endswith(".pcd"):
            cloud = read_pcd_with_viewpoint(cloud_path)
        elif cloud_path.endswith(".ply"):
            cloud = o3d.io.read_point_cloud(cloud_path)
        else:
            raise ValueError(f"Unsupported file format: {cloud_path}")
        final_cloud += cloud

    print(f"Downsampling to {downsample_voxel_size}m ...")
    final_cloud = final_cloud.voxel_down_sample(voxel_size=downsample_voxel_size)
    print(f"Saving merged cloud to {output_cloud_path} ...")
    o3d.io.write_point_cloud(str(output_cloud_path), final_cloud)
    return final_cloud


def merge_downsample_vilens_slam_clouds(vilens_slam_clouds_folder, downsample_voxel_size=0.05, output_cloud_path=None):
    cloud_paths = list(Path(vilens_slam_clouds_folder).rglob("*.pcd"))
    if not output_cloud_path:
        output_cloud_path = Path(vilens_slam_clouds_folder).parent / f"merged_{downsample_voxel_size}m.pcd"
    return merge_downsample_clouds(cloud_paths, output_cloud_path, downsample_voxel_size)


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


def convert_e57_to_pcd(e57_file_path, pcd_file_path, check_output=True, pcd_lib="pypcd4"):
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
    has_intensity = "intensity" in header.point_fields
    # Get the first point cloud (assuming the E57 file contains at least one)
    data = e57_file.read_scan(0, intensity=has_intensity, colors=has_colour)

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

    if pcd_lib == "open3d":
        # cannot save intensity to pcd file using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_sensor_frame)
        # pcd.points = o3d.utility.Vector3dVector(points_np)
        if has_colour:
            pcd.colors = o3d.utility.Vector3dVector(colours / 255)
        o3d.io.write_point_cloud(pcd_file_path, pcd)
        print(f"PCD file saved to {pcd_file_path}")
        modify_pcd_viewpoint(pcd_file_path, pcd_file_path, viewpoint)
    elif pcd_lib == "pypcd4":
        # supported fields: x, y, z, rgb, intensity
        fields = ["x", "y", "z"]
        types = [np.float32, np.float32, np.float32]

        pcd_data = points_sensor_frame

        if has_colour:
            fields += ["rgb"]
            types += [np.float32]
            encoded_colour = PointCloud.encode_rgb(colours)
            encoded_colour = encoded_colour.reshape(-1, 1)
            pcd_data = np.hstack((pcd_data, encoded_colour))
        #     pcd_data = np.hstack((pcd_data, colours / 255))

        if has_intensity:
            fields += ["intensity"]
            types += [np.float32]
            pcd_data = np.hstack((pcd_data, data["intensity"].reshape(-1, 1)))

        fields = tuple(fields)
        types = tuple(types)
        pcd = PointCloud.from_points(pcd_data, fields, types)
        pcd.metadata.viewpoint = tuple(viewpoint)
        pcd.save(pcd_file_path)
    else:
        raise ValueError(f"Unsupported pcd library: {pcd_lib}")

    if check_output:
        saved_cloud = read_pcd_with_viewpoint(pcd_file_path)
        saved_cloud_np = np.array(saved_cloud.points)
        assert np.allclose(saved_cloud_np, points_np, rtol=1e-5, atol=1e-5)
        if has_colour:
            colours_np = np.array(saved_cloud.colors)
            assert np.allclose(colours_np, colours / 255, rtol=1e-5, atol=1e-8)


def transform_cloud_with_se3(cloud_file, se3_matrix, output_cloud_file):
    assert str(cloud_file).endswith(".pcd") or str(cloud_file).endswith(".ply")
    assert is_se3_matrix(se3_matrix)[0], is_se3_matrix(se3_matrix)[1]
    assert str(output_cloud_file).endswith(".pcd") or str(output_cloud_file).endswith(".ply")
    cloud = o3d.io.read_point_cloud(str(cloud_file))
    cloud.transform(se3_matrix)
    o3d.io.write_point_cloud(str(output_cloud_file), cloud)
    logger.info(f"Transformed point cloud with SE(3) and saved as {output_cloud_file}")
