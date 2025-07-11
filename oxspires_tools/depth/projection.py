#!/usr/bin/env python3
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from oxspires_tools.utils import get_pcd


def filter_points_outside_fov(points_in_3d: np.ndarray, fov_deg: float) -> np.ndarray:
    # Remove points outside fov since v2.fisheye.projectPoints doesn't work propery outside fov
    unit_dirs = points_in_3d / np.linalg.norm(points_in_3d, axis=1)[:, np.newaxis]
    mask_fov = np.arccos(unit_dirs[:, 2]) < np.deg2rad(fov_deg / 2)
    return mask_fov


def get_in_image_mask(points_on_img: np.ndarray, w: int, h: int) -> np.ndarray:
    # Get crop mask to remove points outside images
    x = points_on_img[:, 0]
    y = points_on_img[:, 1]
    x_mask = (0 <= x) & (x < w)
    y_mask = (0 <= y) & (y < h)
    valid_mask = x_mask & y_mask
    return valid_mask


def project_points_on_image(
    points_in_3d: np.ndarray,  # (N, 3) points in 3D space
    K: np.ndarray,  # (3, 3) intrinsic camera matrix
    D: np.ndarray,  # (4,) distortion coefficients
    w: int,
    h: int,
    camera_model: str = "OPENCV_FISHEYE",
) -> Tuple[np.ndarray, np.ndarray]:
    rvec = np.zeros((1, 1, 3))
    tvec = np.zeros((1, 1, 3))
    if camera_model == "OPENCV_FISHEYE":
        points_on_img, _ = cv2.fisheye.projectPoints(points_in_3d[np.newaxis], rvec, tvec, K, D)
    elif camera_model == "OPENCV":
        points_on_img, _ = cv2.projectPoints(points_in_3d[np.newaxis], rvec, tvec, K, D)
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")
    if points_on_img is None:
        return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(points_in_3d.shape[0], dtype=bool)
    points_on_img = points_on_img.squeeze()
    if points_on_img.ndim == 1:
        points_on_img = points_on_img[np.newaxis]
    points_on_img = np.round(points_on_img)

    valid_mask = get_in_image_mask(points_on_img, w, h)
    valid_points_on_img = points_on_img[valid_mask]
    return valid_points_on_img, valid_mask


def encode_points_as_depthmap(
    points_on_img: np.ndarray,  # (N, 2)
    points_in_3d: np.ndarray,  # (N, 3)
    h: int,
    w: int,
    is_euclidean: bool,  # If True, depth is L2 distance between point and camera, else z value
    depth_encode_factor: float,  # Scaling factor for UINT16 depth encoding.
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract depth
    if is_euclidean:
        # Depth is L2 distance between point and camera
        depth = np.linalg.norm(points_in_3d, axis=1)
    else:
        # Depth is z value
        depth = points_in_3d[:, 2]
    # Later depth is saved as 16 bit png (0 - 65,535)
    # Remove outside 16 bit range
    z_mask = (depth * depth_encode_factor) < np.iinfo(np.uint16).max

    if not z_mask.all():
        print("[warn] Depth values are too large to be encoded as 16-bit unsigned integers.")

    # Extract only valid value
    valid_points_on_img = points_on_img[z_mask]
    valid_depth = depth[z_mask]
    # valid_points_in_3d = points_in_3d[z_mask]

    depthmap = np.zeros((h, w), dtype=np.uint16)
    u, v = valid_points_on_img.transpose().astype(int)
    sorted_indices = np.argsort(valid_depth)[::-1]
    u = u[sorted_indices]
    v = v[sorted_indices]
    valid_depth = valid_depth[sorted_indices]
    z = (depth_encode_factor * valid_depth).astype(np.uint16)

    # size-aware rendering
    # point_size = 1.0
    # point_radii = point_size / valid_depth
    # unique_x, unique_y, unique_z = size_aware_rendering(u, v, z, point_radii, h, w)
    # depthmap[unique_y, unique_x] = unique_z

    depthmap[v, u] = z
    return depthmap, z_mask, u, v, sorted_indices


def decode_points_from_depthmap(
    depth: np.ndarray,  # (h, w)
    K: np.ndarray,  # (3, 3) intrinsic camera matrix
    D: np.ndarray,  # (4,) distortion coefficients
    is_euclidean: bool,  # If True, depth is L2 distance between point and camera, else z value
    depth_encode_factor: float,  # Scaling factor for UINT16 depth encoding.
    camera_model: str = "OPENCV_FISHEYE",
    image: Optional[np.ndarray] = None,  # (h, w, 3) color image corresponding to the depth map
) -> o3d.geometry.PointCloud:
    # png uint16 encoding value -> m
    depth = depth.astype(float) / depth_encode_factor
    h, w = depth.shape

    # Valid depth mask
    mask = depth > 0
    valid_depth = depth[mask]

    # Get coordinates of valid depth
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x[mask]
    grid_y = grid_y[mask]

    # Project points on image into 3d world coordinates
    points_on_img = np.stack((grid_x, grid_y), axis=1)
    # Adjust shape and type for cv2.fisheye.undistortPoints
    points_on_img = points_on_img[np.newaxis].astype(float)
    # Project points on image onto z = 1 plane
    if camera_model == "OPENCV_FISHEYE":
        points_in_3d = cv2.fisheye.undistortPoints(points_on_img, K, D, P=np.eye(3))
    elif camera_model == "OPENCV":
        points_in_3d = cv2.undistortPoints(points_on_img, K, D, P=np.eye(3))
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")
    if points_in_3d is None:
        return o3d.geometry.PointCloud()
    points_in_3d = points_in_3d.squeeze()
    # Add z value which is one
    # TODO: how to handle one point properly?
    if points_in_3d.ndim == 1:
        points_in_3d = points_in_3d[np.newaxis]

    points_in_3d = np.concatenate((points_in_3d, np.ones((len(points_in_3d), 1))), axis=1)

    if is_euclidean:
        # Normalize since depth is range data (L2 distance between camera and point)
        points_in_3d /= np.linalg.norm(points_in_3d, axis=1, keepdims=True)
    # Multiple by depth
    points_in_3d = valid_depth[:, np.newaxis] * points_in_3d

    # Convert into point cloud
    if image is not None:
        colors = image[mask]
    else:
        colors = None
    pcd = get_pcd(points=points_in_3d, colors=colors)

    return pcd
