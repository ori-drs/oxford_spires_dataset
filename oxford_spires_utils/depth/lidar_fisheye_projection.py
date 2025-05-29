#!/usr/bin/env python3
import shutil
from pathlib import Path

import numpy as np
from nerf_data_pipeline.dataset.utils import get_accumulated_pcd, get_image_pcd_sync_pair
from nerf_data_pipeline.depth.projection import encode_points_as_depthmap, project_pcd_on_image
from nerf_data_pipeline.depth.utils import apply_hidden_point_removal, load_fnt_params, save_projection_outputs
from nerf_data_pipeline.depth.visualize_depthmap_pose import get_transforms
from tqdm.auto import tqdm


def get_depth_from_cloud(
    point_cloud,
    K,
    D,  # distortion coefficients, depending on camera model
    w,
    h,
    fov_deg,
    camera_model,  # "OPENCV_FISHEYE" or "OPENCV"
    depth_factor: float = 256.0,  # depth encoding factor: (depth*depth_encode_factor).astype(np.uint16)"
    is_euclidean=False,  # True: L2 distance between points and camera, False: z_value
    compute_cloud_mask=False,
):
    visible_pcd, hpr_mask = apply_hidden_point_removal(point_cloud)
    points_on_img, points_in_3d, valid_normals, proj_mask = project_pcd_on_image(
        visible_pcd, K, D, w, h, fov_deg, camera_model
    )
    depthmap, normalmap = encode_points_as_depthmap(
        points_on_img,
        points_in_3d,
        h,
        w,
        is_euclidean,
        depth_factor,
        point_size=1,
        normals=valid_normals,
        K=K,
        D=D,
    )
    if compute_cloud_mask:
        cloud_mask = np.zeros(len(point_cloud.points), dtype=bool)
        # proj_mask is for pcd.select_by_index(pt_map) points which are ordered as in hpr_mask, so sort hpr_mask
        hpr_mask = np.array(sorted(hpr_mask))
        filtered_indices = hpr_mask[proj_mask]
        cloud_mask[filtered_indices] = True
        return depthmap, normalmap, cloud_mask
    return depthmap, normalmap


def project_lidar_to_fisheye(
    sensor,
    project_dir: str,
    depth_dir: str,
    normal_dir: str,
    camera_topics_labelled: dict,
    image_folder_path: str,
    image_folder_name: str,
    depth_pose_path: str,
    depth_pose_format: str,
    slam_individual_clouds_new_path: str,
    is_euclidean: bool,
    accum_length: int,
    max_time_diff_camera_and_pose: float,
    image_ext: str = ".jpg",  # image extension
    depth_factor: float = 256.0,  # depth encoding factor: (depth*depth_encode_factor).astype(np.uint16)",
    save_overlay: bool = True,
    save_normal_map: bool = True,
    pose_scale_factor: float = 1.0,  # not 1 if colmap pose
    camera_model: str = "OPENCV_FISHEYE",
    T_cam_base_overwrite: np.ndarray = {},
):
    # args = get_args()
    # Project dir
    proj_dir = Path(project_dir).expanduser()

    if is_euclidean:
        print("Depth is euclidean: L2 distance between points and camera")
    else:
        print("Depth is not euclidean: z_value")
    # Path settings
    # pcd_dir = proj_dir / "individual_clouds"
    # image_dir = proj_dir / "images"
    overlay_dir = proj_dir / (depth_dir + "_overlay")
    output_depth_dir = proj_dir / depth_dir
    if output_depth_dir.exists():
        shutil.rmtree(output_depth_dir)
    if save_overlay and overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    output_depth_dir.mkdir(parents=True)

    if save_normal_map:
        output_normal_dir = proj_dir / normal_dir
        if output_normal_dir.exists():
            shutil.rmtree(output_normal_dir)
        output_normal_dir.mkdir(parents=True)

    image_ext = image_ext if image_ext[0] == "." else f".{image_ext}"
    cam_subdirs = list(camera_topics_labelled.values())
    target_cam_names = list(camera_topics_labelled.keys())

    # Loop over each camera
    for subdir, cam_name in zip(cam_subdirs, target_cam_names):
        print(f"Processing {cam_name} in {subdir} ...")
        # Transformation from base to camera (clouds in individual_clouds are base coordinates)
        # T_cam_lidar = np.linalg.inv(Ts[f"base_{cam_name}"]) @ Ts[f"base_{lidar_name}"]
        K, D, h, w, T_base_cam, fov_deg = load_fnt_params(cam_name, depth_pose_format, depth_pose_path, sensor)
        if cam_name in T_cam_base_overwrite.keys():
            # use the overwriten T_cam_base using T_cam_lidar_overwrite
            T_cam_base = T_cam_base_overwrite[cam_name]
        else:
            # just use the default T_cam_base
            T_cam_base = np.linalg.inv(T_base_cam)

        # Setup input dir
        target_image_subdir = image_folder_path / subdir
        target_depth_subdir = output_depth_dir / subdir
        target_depth_subdir.mkdir(parents=True, exist_ok=True)
        if save_normal_map:
            target_normal_subdir = output_normal_dir / subdir
            target_normal_subdir.mkdir(parents=True, exist_ok=True)
        if save_overlay:
            target_overlay_subdir = overlay_dir / subdir
            target_overlay_subdir.mkdir(parents=True, exist_ok=True)

        # Project lidar points on images and  save as 16 bit depth
        print(f"Target image directory: {target_image_subdir}")
        print(f"Target depth directory: {target_depth_subdir}")

        # Get image and pcd sync pairs
        image_pcd_pairs = get_image_pcd_sync_pair(
            target_image_subdir,
            slam_individual_clouds_new_path,
            image_ext,
            max_time_diff_camera_and_pose,
        )

        # Loop for one camera
        print(f"Fov: {fov_deg}")
        # load all lidar poses as T_WB, so assume vilens-processed lidar in base frame
        T_WBs = get_transforms(
            depth_pose_path,
            depth_pose_format,
            pose_scale_factor,
            T_base_cam,
            subdir,
            image_folder_name,
            frame="base",
            visualise=False,
        )
        for image_path, pcd_path, diff in tqdm(image_pcd_pairs):
            # Load pointclouds
            # pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
            pcd = get_accumulated_pcd(
                pcd_path,
                T_WBs,
                accumulation_length=accum_length,
                max_time_diff_camera_and_pose=max_time_diff_camera_and_pose,
            )
            if pcd is None:
                if accum_length == 0:
                    print(f"{pcd_path} pose not found")
                    continue
                else:
                    raise RuntimeError(f"{pcd_path} pose not found")
            # check if point cloud is empty
            if len(pcd.points) == 0:
                print(f"{pcd_path} is empty")
                continue
            # Transform points into camera coordinates
            pcd.transform(T_cam_base)  # transform lidar points from base to camera frame
            depth, normal = get_depth_from_cloud(pcd, K, D, w, h, fov_deg, camera_model, depth_factor, is_euclidean)
            save_projection_outputs(
                depth,
                normal,
                image_path,
                save_depth_path=(target_depth_subdir / (image_path.stem + ".png")).as_posix(),
                save_normal_path=(target_normal_subdir / (image_path.stem + ".png")).as_posix(),
                save_overlay_path=(target_overlay_subdir / (image_path.stem + ".jpg")).as_posix(),
            )

            # Validation process: Decode points from depthmap and compare with pcd inside fov. Slower the loop
            # if use_pcd_validation:
            #     pcd_decode = decode_points_from_depthmap(depthmap, K, D, is_euclidean, depth_factor, camera_model)
            #     pcd_inside_fov = get_pcd(points_in_3d)
            #     diff_in_3d = pcd_decode.compute_point_cloud_distance(pcd_inside_fov)
            #     # print(np.asarray(diff_in_3d).max())
            #     if np.asarray(diff_in_3d).max() > 0.5:
            #         print(f"Warning. compute_point_cloud_distance.max()={np.asarray(diff_in_3d).max():.2f}.")
            #         print("Validation process failed.")
            #         print("Please make sure if fov_deg is set property or camera is well calibrated even corner.")
            # # Debug
            # if debug:
            #     pcd_decode = decode_points_from_depthmap(depthmap, K, D, is_euclidean, depth_factor, camera_model)
            #     pcd_inside_fov = get_pcd(points_in_3d)
            #     o3d.io.write_point_cloud(f"cnt{cnt}_original.pcd", pcd)
            #     o3d.io.write_point_cloud(f"cnt{cnt}_inside_fov.pcd", pcd_inside_fov)
            #     o3d.io.write_point_cloud(f"cnt{cnt}_decode.pcd", pcd_decode)
            #     cnt += 1
            #     o3d.visualization.draw_geometries([pcd])
            #     o3d.visualization.draw_geometries([pcd_inside_fov])
            #     o3d.visualization.draw_geometries([pcd_decode])
            #     o3d.visualization.draw_geometries([pcd, pcd_decode])
