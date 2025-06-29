from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def get_overlay(camera_image, depth_image, cmap="hsv", circle_radius=2, circle_thickness=2):
    # Overlay depth on image
    depth_mask = depth_image > 0

    cmap = plt.cm.get_cmap(cmap)
    if depth_image.max() == 0:
        print("WARNING: Depth image is all zeros")
        return camera_image
    depth_norm = depth_image / np.max(depth_image)
    depth_image_cmap = cmap(depth_norm)[:, :, :3]
    depth_image_cmap = (depth_image_cmap * 255).astype(np.uint8)
    depth_image_cmap[~depth_mask] = 0
    depth_image_cmap = cv2.cvtColor(depth_image_cmap, cv2.COLOR_RGB2BGR)

    # overlay = depth_image_cmap

    overlay = camera_image.copy()
    for x, y in zip(*np.where(depth_mask)):
        cv2.circle(
            overlay,
            (y, x),
            circle_radius,
            depth_image_cmap[x, y].tolist(),
            circle_thickness,
        )
    overlay[depth_mask] = depth_image_cmap[depth_mask]
    return overlay


def apply_hidden_point_removal(pcd: o3d.geometry.PointCloud, visualise=False):
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    _, hpr_mask = pcd.hidden_point_removal([0, 0, 0], diameter * 200)
    visible_pcd = pcd.select_by_index(hpr_mask)
    if visualise:
        origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([visible_pcd, origin_axis])

    return visible_pcd, hpr_mask


def get_projection_output_paths(save_name):
    save_depth_path = (Path(target_depth_subdir) / (save_name + ".png")).as_posix() if target_depth_subdir else None
    save_normal_path = (Path(target_normal_subdir) / (save_name + ".png")).as_posix() if target_normal_subdir else None
    save_overlay_path = (
        (Path(target_overlay_subdir) / (save_name + ".jpg")).as_posix() if target_overlay_subdir else None
    )
    return save_depth_path, save_normal_path, save_overlay_path


def save_projection_outputs(
    depthmap,
    normalmap=None,
    image_path=None,
    save_depth_path=None,
    save_normal_path=None,
    save_overlay_path=None,
):
    camera_image = cv2.imread(Path(image_path).as_posix()) if image_path else None

    if save_depth_path:
        cv2.imwrite(str(save_depth_path), depthmap)
    if save_normal_path and normalmap is not None:
        normalmap = cv2.cvtColor(normalmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_normal_path), normalmap)
    if save_overlay_path:
        assert camera_image is not None
        overlay = get_overlay(camera_image, depthmap)
        cv2.imwrite(str(save_overlay_path), overlay)


def get_K_D_h_w_from_colmap_frame(frame, camera_model, cam_name):
    K = np.zeros((3, 3))
    K[0, 0] = frame["fl_x"]
    K[1, 1] = frame["fl_y"]
    K[0, 2] = frame["cx"]
    K[1, 2] = frame["cy"]
    if camera_model == "OPENCV_FISHEYE":
        D = np.array([frame["k1"], frame["k2"], frame["k3"], frame["k4"]])
    elif camera_model == "OPENCV":
        D = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"]])
    h = frame["h"]
    w = frame["w"]
    print(f"{cam_name} {camera_model}\nK: {K}\nD: {D}\nh: {h}, w: {w}")

    return K, D, h, w
