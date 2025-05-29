from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from nerf_data_pipeline.dataset.utils import find_closest_in_sorted
from nerf_data_pipeline.depth.projection import decode_points_from_depthmap
from nerf_data_pipeline.depth.utils import load_fnt_params
from nerf_data_pipeline.pose.file_interfaces import NeRFTrajReader, VilensSlamTrajReader
from nerf_data_pipeline.pose.pose_convention import PoseConvention
from tqdm import tqdm


def get_transforms(
    robotics_pose_file,
    depth_pose_format,
    scale_factor,
    T_BC,
    camera_topic,
    image_folder_name,
    frame="camera",  # returned pose's frame. base: T_WB; camera: T_WC
    visualise=False,
):
    # Read trajectory
    if depth_pose_format == "vilens_slam":
        reader = VilensSlamTrajReader(robotics_pose_file)
        pose_traj = reader.read_file()
        pose_traj.scale(scale_factor)  # T_WB
        # pose_traj.transform(PoseConvention.get_transform("robotics", "vision"), right_mul=True)
        if frame == "camera":
            pose_traj.transform(T_BC, right_mul=True)  # T_BC * T_WB=T_WC
        # raise NotImplementedError("TODO: fix this")
    elif depth_pose_format == "nerf":
        camera_folder = image_folder_name + "/" + camera_topic
        reader = NeRFTrajReader(robotics_pose_file, camera_folder, nerf_reader_sort_timestamp=True)
        pose_traj = reader.read_file()
        pose_traj.scale(scale_factor)  # T_WC in nerf convention
        pose_traj.transform(PoseConvention.get_transform("nerf", "vision"), right_mul=True)
        if frame == "base":
            T_CB = np.linalg.inv(T_BC)
            pose_traj.transform(T_CB, right_mul=True)  # T_WB

    pcds = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)]
    scaled_Ts = {}
    for key, T in zip(pose_traj.timestamps, pose_traj.poses_se3):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        axis.transform(T)
        pcds.append(axis)
        scaled_Ts[str(key)] = T
    if visualise:
        print("Visualize loaded pose.")
        print("Press ECS to exit.")
        o3d.visualization.draw_geometries(pcds)
    return scaled_Ts


def visualise_depth(
    sensor,
    project_dir: str,
    depth_dir: str,
    image_folder_path: str,
    image_folder_name: str,
    camera_topics_labelled: dict,
    depth_pose_path: str,
    depth_pose_format: str,
    is_euclidean: bool,
    max_time_diff_camera_and_pose: float,
    image_ext: str = ".jpg",  # image extension
    depth_factor: float = 256.0,  # depth encoding factor: (depth*depth_encode_factor).astype(np.uint16)",
    pose_scale_factor: float = 1.0,  # not 1 if colmap pose
    camera_model: str = "OPENCV_FISHEYE",
):
    image_ext = image_ext if image_ext[0] == "." else f".{image_ext}"

    if is_euclidean:
        print("Depth is euclidean: L2 distance between points and camera")
    else:
        print("Depth is not euclidean: z_value")
    assert depth_pose_format in ["nerf", "vilens_slam"]
    for camera_name in sensor.camera_topics_labelled.keys():
        camera_topic = sensor.camera_topics_labelled[camera_name]
        K, D, h, w, T_base_cam, fov_deg = load_fnt_params(camera_name, depth_pose_format, depth_pose_path, sensor)
        # K, D, w, h = fnt_config.load_camera_K_D_w_h(camera_name)

        # Project dir
        proj_dir = Path(project_dir).expanduser()
        verify_image_dir = image_folder_path / camera_topic
        verify_depth_dir = proj_dir / depth_dir / camera_topic
        T_base_cam_target = sensor.tf.get_transform(camera_name, "base")
        scaled_Ts = get_transforms(
            depth_pose_path,
            depth_pose_format,
            pose_scale_factor,
            T_base_cam_target,
            camera_topic,
            image_folder_name,
            frame="camera",
            visualise=False,
        )

        depth_paths = sorted(list(verify_depth_dir.glob("*.png")))
        print(f"{len(depth_paths)} depth images found in {verify_depth_dir}")
        image_paths = []
        for depth_path in depth_paths:
            #! TODO raw/image hard coded
            image_path = Path(str(depth_path).replace(depth_dir, "raw/images").replace(".png", image_ext))
            if image_path.exists():
                image_paths.append(image_path)
            else:
                assert RuntimeError(f"Cannot find corresponding image path: {image_path}")
        print(f"{len(image_paths)}   rgb images found in {verify_image_dir}")

        print("Combining pointclouds")
        cloud_folder = proj_dir / f"{depth_dir}_cloud" / f"{camera_topic}"
        cloud_folder.mkdir(parents=True, exist_ok=True)
        pcd_combined = o3d.geometry.PointCloud()
        valid_pcd = 0
        pose_timestamps_str = list(scaled_Ts.keys())
        pose_timestamps_float = [float(i) for i in pose_timestamps_str]
        pbar = tqdm(total=len(image_paths))
        for image_path, depth_path in zip(image_paths, depth_paths):
            pbar.set_description(f"Processing {image_path.stem}")
            pbar.update(1)
            # Skip if pose is not found
            image_timestamp = float(image_path.stem)
            image_timestamp, diff, idx = find_closest_in_sorted(pose_timestamps_float, image_timestamp)
            if diff > max_time_diff_camera_and_pose:
                continue
            img = cv2.imread(image_path.as_posix())
            # Open3D color is RGB (0 - 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_ANYDEPTH)

            # Decode points from depthmap

            pcd = decode_points_from_depthmap(depth, K, D, is_euclidean, depth_factor, img, camera_model)

            # Apply transformation
            T = scaled_Ts[pose_timestamps_str[idx]]
            # T = scaled_Ts[image_path.stem]
            pcd.transform(T)
            pcd_combined += pcd
            o3d.io.write_point_cloud(
                str(cloud_folder / f"{pose_timestamps_str[idx]}.pcd"),
                pcd,
            )
            valid_pcd += 1
            # pcds.append(pcd)
        pbar.close()
        print(f"Combined {valid_pcd} pointclouds")
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        (proj_dir / f"{depth_dir}_cloud").mkdir(parents=True, exist_ok=True)
        # o3d.io.write_point_cloud(str(proj_dir / f"{args.depth_dir}_cloud" / f"{cam_subdir}.pcd"), pcd_combined)
        # o3d.visualization.draw_geometries([axis, pcd_combined])
