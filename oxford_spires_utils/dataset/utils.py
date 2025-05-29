import re
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


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


def get_pcd(points, colors=None):
    """Get open3d pointcloud from points and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        assert colors.max() <= 1.0 and colors.min() >= 0.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


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


def get_image_depthmap_sync_pair(
    image_dir: Path,
    depthmap_dir: Path,
    image_ext: str = ".jpg",
    depthmap_ext: str = ".png",
    timestamp_threshold: float = 0.01,
) -> List[Tuple[Path, Path, float]]:
    """
    Find image and depthmap pairs that are synchronized based on their timestamps.

    Args:
        image_dir (Path): Directory containing the image files.
        depthmap_dir (Path): Directory containing the depthmap files.
        image_ext (str): File extension of the image files (e.g. ".jpg").
        depthmap_ext (str): File extension of the depthmap files (e.g. ".png").
        timestamp_threshold (float): Maximum time difference allowed between an image and a depthmap, in seconds.
            Defaults to 0.05.

    Returns:
        List[Tuple[Path, Path, float]]: A list of tuples, where each tuple represents an image and a depthmap that are
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
    assert len(image_paths) > 0, f"No images are found in {image_dir}"
    # print(f"Loaded {len(image_paths)} images")

    # Collect all image and depthmap pairs which have timestamp close enough
    image_depthmap_pairs = []
    for it in sorted(list(depthmap_dir.glob(f"*{depthmap_ext}"))):
        ret = re.findall(r"\d+", it.name)
        timestamp = float(".".join(ret))
        image_timestamp, diff, _ = find_closest_in_sorted(image_timestamps, timestamp)
        # print(diff)
        if diff < timestamp_threshold:
            image_depthmap_pairs.append((image_paths[image_timestamp], it, diff))

    # print(f"Found {len(image_depthmap_pairs)} image-pcd pairs")
    return image_depthmap_pairs


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
