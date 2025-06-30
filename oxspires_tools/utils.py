import re
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple

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


def get_pcd(points, colors=None):
    """Get open3d pointcloud from points and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        assert colors.max() <= 1.0 and colors.min() >= 0.0, "Colors should be in range [0, 1]"
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def get_accumulated_pcd(current_pcd, transforms, accumulation_length=0, max_time_diff_camera_and_pose=0.0):
    def get_T_WB_from_timestamp(timestamp, transforms, max_time_diff_camera_and_pose=0.0):
        closest_timestamp = min(transforms.keys(), key=lambda x: abs(float(x) - float(timestamp)))
        diff = abs(float(closest_timestamp) - float(timestamp))
        if diff > max_time_diff_camera_and_pose:
            # print(f"WARNING: {timestamp}'S diff from transforms is {diff} > {max_time_diff_camera_and_pose} ")
            return None
        if max_time_diff_camera_and_pose == 0 and timestamp != closest_timestamp:
            print(f"Warning: closest timestamp is {closest_timestamp} but current timestamp is {timestamp}")
            input("Press Enter to continue...")
        timestamp = closest_timestamp
        return transforms[timestamp]

    # Get pcd directory
    pcd_dir = current_pcd.parent

    pcd_paths = sorted(list(pcd_dir.glob("*pcd")))
    accumulated_pcd_paths = []
    current_idx = pcd_paths.index(current_pcd)
    for i in range(accumulation_length * 2 + 1):
        new_idx = current_idx - accumulation_length + i
        if new_idx < 0 or new_idx >= len(pcd_paths):
            continue
        accumulated_pcd_paths.append(pcd_paths[new_idx])

    accumulated_pcd = o3d.geometry.PointCloud()
    current_timestamp = current_pcd.stem[6:].replace("_", ".")
    # remove tailing zeros
    current_timestamp = re.sub(r"0+$", "", current_timestamp)
    T_WB = get_T_WB_from_timestamp(current_timestamp, transforms, max_time_diff_camera_and_pose)
    if T_WB is None:
        print(f"current_timestamp: {current_timestamp} not found")
        return None
    # print(f"Current timestamp: {current_timestamp}")
    for pcd_path in accumulated_pcd_paths:
        pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
        that_timestamp = pcd_path.stem[6:].replace("_", ".")
        # print(f"that timestamp: {that_timestamp}")
        # remove tailing zeros
        that_timestamp = re.sub(r"0+$", "", that_timestamp)
        # Accumulate pcd if pose exists

        T_WA = get_T_WB_from_timestamp(that_timestamp, transforms, max_time_diff_camera_and_pose)
        if T_WA is None:
            # print(f"skipping {that_timestamp} for {current_timestamp}")
            continue
        T_BA = np.linalg.inv(T_WB) @ T_WA
        accumulated_pcd += pcd.transform(T_BA)
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([axis,accumulated_pcd])
    return accumulated_pcd


def find_closest_in_sorted(array: List, value: float):
    """
    Find and return closest value and difference
    """
    # Over or under any value in the array
    assert isinstance(value, (float, np.float128))
    if array[-1] < value:
        return array[-1], value - array[-1], len(array) - 1
    if array[0] > value:
        return array[0], array[0] - value, 0

    # Find closest value using binary search
    idx = bisect_left(array, value)
    diff_left = abs(value - array[idx - 1])
    diff_right = abs(array[idx] - value)
    if diff_left < diff_right:
        return array[idx - 1], diff_left, idx - 1
    else:
        return array[idx], diff_right, idx


def get_image_pcd_sync_pair(
    image_dir: Path, pcd_dir: Path, image_ext: str, timestamp_threshold: float = 0.05
) -> List[Tuple[Path, Path, float]]:
    """
    Find image and point cloud pairs that are synchronized based on their timestamps.

    Args:
        image_dir (Path): Directory containing the image files.
        pcd_dir (Path): Directory containing the point cloud files.
        image_ext (str): File extension of the image files (e.g. ".jpg").
        timestamp_threshold (float): Maximum time difference allowed between an image and a point cloud, in seconds.
            Defaults to 0.05.

    Returns:
        List[Tuple[Path, Path, float]]: A list of tuples, where each tuple represents an image and a point cloud that are
            synchronized, along with their time difference in seconds.
    """
    # Load images and timestamps
    image_timestamps = []
    image_paths = {}
    for it in sorted(list(image_dir.glob(f"*{image_ext}"))):
        ret = re.findall(r"\d+", it.name)
        timestamp = float(".".join(ret))
        image_timestamps.append(timestamp)
        image_paths[timestamp] = it
    assert len(image_paths) > 0, "No images are found"
    print(f"Loaded {len(image_paths)} images")

    # Collect all image and lidar pair which have timestamp close enough
    image_pcd_pairs = []
    for it in sorted(list(Path(pcd_dir).glob("*pcd"))):
        ret = re.findall(r"\d+", it.name)
        timestamp = float(".".join(ret))
        image_timestamp, diff, _ = find_closest_in_sorted(image_timestamps, timestamp)
        # print(diff)
        if diff < timestamp_threshold:
            image_pcd_pairs.append((image_paths[image_timestamp], it, diff))

    print(f"Found {len(image_pcd_pairs)} image-pcd pairs")
    return image_pcd_pairs
