#!/usr/bin/env python3
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from oxford_spires_utils.dataset.utils import get_pcd


def project_pcd_on_image(
    pcd: o3d.geometry.PointCloud,
    K: np.ndarray,
    D: np.ndarray,
    w: int,
    h: int,
    fov_deg: float,
    camera_model: str = "OPENCV_FISHEYE",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a 3D point cloud onto a 2D image plane using the fisheye camera model.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to be projected.
        K (np.ndarray): The intrinsic camera matrix of shape (3, 3).
        D (np.ndarray): The distortion coefficients of shape (4,).
        w (int): The width of the output image in pixels.
        h (int): The height of the output image in pixels.
        fov_deg (float): The diagonal field of view of the camera in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays representing the projected points and their corresponding
            3D coordinates in the point cloud. The first array has shape (N, 2), where N is the number of points projected
            onto the image plane, and each row contains the (x, y) coordinates of a projected point. The second array has
            shape (N, 3), where each row contains the 3D coordinates of the corresponding point in the input point cloud.

    Notes:
        The input point cloud is assumed to be in the camera coordinate system, with the camera at the origin and pointing
        in the positive z direction. The output image has the origin at the top-left corner and the x-axis pointing to the
        right and the y-axis pointing down.

        This function uses the OpenCV fisheye camera model to project the points onto the image plane. Points outside the
        camera field of view are discarded. The output points are rounded to the nearest integer pixel coordinates.

    """
    # Get points
    points_in_3d = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # Remove points outside fov
    # cv2.fisheye.projectPoints doesn't work propery outside fov
    # Normalized direction vector
    unit_dirs = points_in_3d / np.linalg.norm(points_in_3d, axis=1)[:, np.newaxis]
    # arccos(unit_dirs.dot(z_axis)) < np.deg2rad(fov_deg/2)
    mask_fov = np.arccos(unit_dirs[:, 2]) < np.deg2rad(fov_deg / 2)
    points_in_3d = points_in_3d[mask_fov]
    normals = normals[mask_fov] if normals.size > 0 else normals

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

    # Get crop mask to remove points outside images
    x = points_on_img[:, 0]
    y = points_on_img[:, 1]
    x_mask = (0 <= x) & (x < w)
    y_mask = (0 <= y) & (y < h)
    valid_mask = x_mask & y_mask

    valid_points_on_img = points_on_img[valid_mask]
    valid_points_in_3d = points_in_3d[valid_mask]
    valid_normals = normals[valid_mask] if normals.size > 0 else normals
    # combine mask_fov and valid_mask
    indices_fov_mask = np.where(mask_fov)[0]
    full_mask = np.zeros(len(mask_fov), dtype=bool)
    full_mask[indices_fov_mask] = valid_mask
    return valid_points_on_img, valid_points_in_3d, valid_normals, full_mask


def encode_points_as_depthmap(
    points_on_img: np.ndarray,
    points_in_3d: np.ndarray,
    h: int,
    w: int,
    is_euclidean: bool,
    depth_encode_factor: float,
    point_size: float = 1.0,
    normals: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Encode a set of 3D points as a depth map. It is assumed to be called after `project_pcd_on_image`.

    Args:
        points_on_img (np.ndarray): A numpy array of shape (N, 2) representing the (x, y) coordinates of the projected 3D
            points on the image plane.
        points_in_3d (np.ndarray): A numpy array of shape (N, 3) representing the corresponding 3D coordinates of the
            projected points in the input point cloud.
        h (int): The height of the output depth map in pixels.
        w (int): The width of the output depth map in pixels.
        is_euclidean (bool): A flag indicating whether to use the Euclidean distance between the points and the camera
            as the depth value. If False, the z coordinate of each point is used instead.
        depth_encode_factor (float): A scaling factor applied to the depth values to convert them to 16-bit unsigned
            integers. The maximum depth value that can be represented is (2^16 - 1) / depth_encode_factor.

    Returns:
        np.ndarray: A 2D numpy array of shape (h, w) representing the encoded depth map. Each pixel value is a 16-bit
            unsigned integer representing the depth value of the corresponding point in the input point cloud. Pixels
            without any corresponding point are set to 0.
    """
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
    # point_radii = point_size / valid_depth
    # unique_x, unique_y, unique_z = size_aware_rendering(u, v, z, point_radii, h, w)
    # depthmap[unique_y, unique_x] = unique_z

    depthmap[v, u] = z

    if normals is not None and normals.size > 0:
        assert normals.shape[0] == points_in_3d.shape[0]
        assert K is not None and D is not None, "Intrinsics needed to flip normals"
        valid_normals = normals[z_mask]
        valid_normals = valid_normals[sorted_indices]
        normalmap = compute_normalmap(valid_normals, v, u, h, w, K, D)
        return depthmap, normalmap
    return depthmap, None


def decode_points_from_depthmap(
    depth: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    is_euclidean: bool,
    depth_encode_factor: float,
    image: Optional[np.ndarray] = None,
    camera_model: str = "OPENCV_FISHEYE",
) -> o3d.geometry.PointCloud:
    """
    Decode a depth map into a set of 3D points.

    Args:
        depth (np.ndarray): A 2D numpy array of shape (h, w) representing the input depth map. Each pixel value is a
            16-bit unsigned integer representing the depth value of the corresponding point in the 3D point cloud.
        K (np.ndarray): The intrinsic camera matrix of shape (3, 3).
        D (np.ndarray): The distortion coefficients of shape (4,).
        is_euclidean (bool): A flag indicating whether the input depth values are Euclidean distances between the camera
            and each point. If False, the depth values represent the z-coordinate of each point in the camera coordinate
            system.
        depth_encode_factor (float): A scaling factor applied to the depth values to convert them from 16-bit unsigned
            integers to floating point numbers. The depth values are scaled by this factor before being decoded into 3D
            points.
        image (np.ndarray, optional): A 2D numpy array of shape (h, w, 3) representing the color image corresponding to
            the input depth map. If None (default), the output point cloud will not have colors.

    Returns:
        o3d.geometry.PointCloud: An instance of the Open3D PointCloud class representing the 3D point cloud decoded from
            the input depth map. If `image` is not None, each point is colored according to the corresponding pixel in
            the input color image. If `image` is None, the returned point cloud will not have colors.
    """
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


def compute_normalmap(normals, v, u, h, w, K, D):
    normalmap = np.zeros((h, w, 3), dtype=np.float32)
    if normals.size == 0:
        return normalmap
    assert normals.max() <= 1.0 + 1e-5, normals.max()
    assert normals.min() >= -1.0 - 1e-5, normals.min()

    normalmap[v, u] = normals

    # flip if normal is pointing away from camera
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    x = (i - cx) / fx
    y = (j - cy) / fy
    z = np.ones_like(x)

    pixel_vectors = np.stack((x, y, z), axis=-1)
    cos_theta = np.sum(pixel_vectors * normalmap, axis=-1)
    normalmap[cos_theta > 0] *= -1
    assert normals.max() <= 1.0 + 1e-5, normals.max()
    assert normals.min() >= -1.0 - 1e-5, normals.min()
    normalised_normalmap = ((normalmap + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # hard code empty normal to be 128,128,128
    old_point = np.array([127, 127, 127], dtype=np.uint8)
    new_point = np.array([128, 128, 128], dtype=np.uint8)
    indices = np.where(np.all(normalised_normalmap == old_point, axis=-1))
    normalised_normalmap[indices] = new_point

    return normalised_normalmap


def size_aware_rendering(u, v, z, point_radii, h, w):
    inflated_x, inflated_y, inflated_z = assign_radius_values_new(u, v, z, point_radii, h, w)
    unique_x, unique_y, unique_z = select_minimum_values(inflated_x, inflated_y, inflated_z)
    return unique_x, unique_y, unique_z


def assign_radius_values_new(x_coords, y_coords, values, radii, h, w):
    neighbors_x = np.array([])
    neighbors_y = np.array([])
    neighbors_values = np.array([])

    # Create a grid of integer coordinates
    r_int = np.floor(max(radii)).astype(int)  # enlarged radius to include all points within the circle
    grid_x, grid_y = np.meshgrid(np.arange(-r_int, r_int + 1), np.arange(-r_int, r_int + 1))
    grid_x = np.tile(grid_x, (len(x_coords), 1, 1))
    grid_y = np.tile(grid_y, (len(y_coords), 1, 1))

    # Create a mask of points within the circular region
    mask = grid_x**2 + grid_y**2 <= radii[..., None, None] ** 2  # 0,0 always included even if r<1

    # Create arrays for each coordinate and value
    neighbors_x = x_coords[..., None, None] + grid_x
    neighbors_x = neighbors_x[mask]

    neighbors_y = y_coords[..., None, None] + grid_y
    neighbors_y = neighbors_y[mask]

    neighbors_values = np.repeat(values[..., None, None], grid_x.shape[1], axis=1)
    neighbors_values = np.repeat(neighbors_values, grid_y.shape[2], axis=2)
    neighbors_values = neighbors_values[mask]

    valid_xy = (neighbors_x >= 0) & (neighbors_x < w) & (neighbors_y >= 0) & (neighbors_y < h)
    neighbors_x = neighbors_x[valid_xy]
    neighbors_y = neighbors_y[valid_xy]
    neighbors_values = neighbors_values[valid_xy]

    return neighbors_x, neighbors_y, neighbors_values


def select_minimum_values(x_coords, y_coords, old_value):
    # Create a combined array of (x, y, value)
    combined = np.column_stack((x_coords, y_coords, old_value))

    # Sort the combined array based on x, y, and value
    combined_sorted = combined[combined[:, 2].argsort(kind="stable")]

    # Find unique (x, y) coordinates
    unique_coords, unique_indices = np.unique(combined_sorted[:, :2], axis=0, return_index=True)

    # Create a result array with the same shape as the input arrays
    new_combined = combined_sorted[unique_indices]

    new_x_coords = new_combined[:, 0]
    new_y_coords = new_combined[:, 1]
    new_depth = new_combined[:, 2]

    return new_x_coords, new_y_coords, new_depth
