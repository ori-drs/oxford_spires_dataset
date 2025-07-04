#!/usr/bin/env python3
import shutil
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

from oxspires_tools.depth.main import get_depth_from_cloud
from oxspires_tools.depth.utils import save_projection_outputs
from oxspires_tools.sensor import Sensor
from oxspires_tools.utils import get_accumulated_pcd, get_image_pcd_sync_pair, get_transforms


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
):
    proj_dir = Path(project_dir).expanduser()

    if is_euclidean:
        print("Depth is euclidean: L2 distance between points and camera")
    else:
        print("Depth is not euclidean: z_value")

    output_depth_dir = proj_dir / depth_dir
    if output_depth_dir.exists():
        shutil.rmtree(output_depth_dir)
    output_depth_dir.mkdir(parents=True)

    overlay_dir = proj_dir / (depth_dir + "_overlay")
    if save_overlay and overlay_dir.exists():
        shutil.rmtree(overlay_dir)

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
        K, D, h, w, fov_deg, _ = sensor.get_params_for_depth(cam_name, depth_pose_format, depth_pose_path)
        T_cam_base = sensor.tf.get_transform("base", cam_name)
        T_base_cam = np.linalg.inv(T_cam_base)  # T_BC

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


if __name__ == "__main__":
    config_yaml_path = Path("/home/docker_dev/oxford_spires_dataset/config/sensor.yaml")
    with open(config_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    sensor = Sensor(**yaml_data["sensor"])

    project_lidar_to_fisheye(
        sensor=sensor,
        project_dir="/home/docker_dev/oxford_spires_dataset/data/raw/hf_test/sequences/2024-03-18-christ-church-01/ndp_output",
        depth_dir="processed/depths_euc_accum_0",
        normal_dir="processed/normals_euc_accum_0",
        camera_topics_labelled=sensor.camera_topics_labelled,
        image_folder_path=Path("/home/docker_dev/oxford_spires_dataset/data/raw/hf_test/sequences/2024-03-18-christ-church-01/ndp_output/raw/images"),
        image_folder_name="images",
        depth_pose_path=Path("/home/docker_dev/oxford_spires_dataset/data/raw/hf_test/sequences/2024-03-18-christ-church-01/ndp_output/processed/output_colmap/transforms_colmap_scaled.json"),
        depth_pose_format="nerf",
        slam_individual_clouds_new_path=Path("/home/docker_dev/oxford_spires_dataset/data/raw/hf_test/sequences/2024-03-18-christ-church-01/ndp_output/raw/undist-clouds"),
        is_euclidean=True,
        accum_length=0,
        max_time_diff_camera_and_pose=0.025,
        image_ext=".jpg",  # image extension
        depth_factor=256.0,  # depth encoding factor: (depth*depth_encode_factor).astype(np.uint16)
        save_overlay=True,
        save_normal_map=True,
        pose_scale_factor=1.0,  # not 1 if colmap pose
        camera_model="OPENCV_FISHEYE",
    )  # fmt: skip
