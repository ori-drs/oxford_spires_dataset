from typing import Tuple, Union

import numpy as np
import open3d as o3d

from oxspires_tools.depth.projection import (
    encode_points_as_depthmap,
    filter_points_outside_fov,
    project_points_on_image,
)
from oxspires_tools.depth.surface_normal import compute_normalmap
from oxspires_tools.depth.utils import apply_hidden_point_removal


def get_depth_from_cloud(
    point_cloud: o3d.geometry.PointCloud,
    K: np.ndarray,  # (3, 3) intrinsic camera matrix
    D: np.ndarray,  # (4,) distortion coefficients
    w: int,
    h: int,
    fov_deg: float,  # field of view in degrees
    camera_model: str,  # "OPENCV_FISHEYE" or "OPENCV"
    depth_factor: float = 256.0,  # depth encoding factor: (depth*depth_encode_factor).astype(np.uint16)"
    is_euclidean: bool = False,  # True: L2 distance between points and camera, False: z_value
    compute_cloud_mask=False,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    # 1. Hidden Point Removal
    visible_pcd, hpr_mask = apply_hidden_point_removal(point_cloud)
    visible_cloud_np = np.asarray(visible_pcd.points)
    normals_np = np.asarray(visible_pcd.normals)
    # 2. FOV filtering
    mask_fov = filter_points_outside_fov(visible_cloud_np, fov_deg)
    visible_cloud_np = visible_cloud_np[mask_fov]
    # 3. Projection
    points_on_img, in_image_mask = project_points_on_image(visible_cloud_np, K, D, w, h, camera_model)
    points_in_3d = visible_cloud_np[in_image_mask]
    # 4. Encode points as UINT16 depth IMAGE
    depthmap, z_mask, u, v, sorted_indices = encode_points_as_depthmap(
        points_on_img, points_in_3d, h, w, is_euclidean, depth_factor
    )
    if normals_np.size > 0:
        valid_normals_np = normals_np[mask_fov][in_image_mask][z_mask]
        valid_normals_np = valid_normals_np[sorted_indices]
        assert valid_normals_np.shape[0] == points_on_img.shape[0]
        normalmap = compute_normalmap(valid_normals_np, v, u, h, w, K, D)
    if compute_cloud_mask:
        cloud_mask = np.zeros(len(point_cloud.points), dtype=bool)
        # proj_mask is for pcd.select_by_index(pt_map) points which are ordered as in hpr_mask, so sort hpr_mask
        hpr_mask = np.array(sorted(hpr_mask))

        indices_fov_mask = np.where(mask_fov)[0]
        full_mask = np.zeros(len(mask_fov), dtype=bool)
        full_mask[indices_fov_mask] = in_image_mask
        filtered_indices = hpr_mask[full_mask]
        cloud_mask[filtered_indices] = True
        return depthmap, normalmap, cloud_mask
    else:
        return depthmap, normalmap
