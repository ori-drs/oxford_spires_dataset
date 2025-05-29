# import rospkg
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))


def process_lidar(cloud_msg, topic, timestamp, save_folder):
    import sensor_msgs.point_cloud2 as pc2

    cloud = pc2.read_points_list(cloud_msg)
    cloud_array = np.array(cloud)
    cloud_array_xyz = cloud_array[:, :3]
    # save array
    save_path = Path(save_folder) / f"{timestamp.to_sec():.9f}.npy"
    np.save(save_path, cloud_array_xyz)


def process_image(image, topic, t, save_folder):
    save_path = Path(save_folder) / f"{t.to_sec():.9f}.jpg"
    cv2.imwrite(str(save_path), image)


def load_lidar_camera_pairs_from_bag(rosbag_path, lidar_topic, camera_topic, save_folder, end_timestamp=99999999999999):
    sys.path.append(str(Path(__file__).absolute().parent.parent.parent / "scripts" / "raw_data_handler"))
    import rospy
    from nerf_data_pipeline.ros.rosbag_utils import ROSBag

    if not isinstance(end_timestamp, rospy.Time):
        end_timestamp = rospy.Time.from_sec(end_timestamp)

    frontier_bag = ROSBag(rosbag_path)
    frontier_bag.load_msgs(
        lidar_topic, process_lidar, read_every_n_sec=10, save_folder=save_folder, end_timestamp=end_timestamp
    )
    saved_npys = list(Path(save_folder).glob("*.npy"))
    target_timestamp_list = [float(npy.stem) for npy in saved_npys]
    frontier_bag.load_msgs(
        camera_topic,
        process_image,
        read_every_n_sec=100000,
        target_timestamp_list=target_timestamp_list,
        max_diff=0.05,
        save_folder=save_folder,
        end_timestamp=end_timestamp,
    )


def run_overlay(saved_npys, saved_images, save_folder, config_yaml_path, T_cam0_lidar=None):
    import open3d as o3d
    from nerf_data_pipeline.config.config_class import Config
    from nerf_data_pipeline.depth.lidar_fisheye_projection import get_depth_from_cloud
    from nerf_data_pipeline.depth.utils import load_fnt_params, save_projection_outputs

    with open(config_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    config = Config(yaml_data)
    K, D, h, w, T_base_cam, fov_deg = load_fnt_params(
        "cam_front", depth_pose_format, config.depth.depth_pose_path, config.sensor
    )
    camera_model = config.sensor.camera_model
    if T_cam0_lidar is None:
        T_cam0_lidar = config.sensor.tf.get_transform("lidar", "cam_right")
    for cloud in saved_npys:
        cloud_timestamp = float(cloud.stem)
        nearest_image = min(saved_images, key=lambda x: abs(float(x.stem) - cloud_timestamp))
        print(cloud, nearest_image)
        cloud_array = np.load(cloud)
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_array)
        o3d_cloud.transform(T_cam0_lidar)
        depthmap, normalmap = get_depth_from_cloud(
            o3d_cloud,
            K,
            D,
            w,
            h,
            fov_deg,
            camera_model,
            256.0,
        )
        save_projection_outputs(
            depthmap, image_path=nearest_image, save_overlay_path=save_folder / "overlay" / f"{cloud.stem}.jpg"
        )


rosbag_path = "/home/yifu/data/nerf_data_pipeline/ori/rosbag/2023-08-21-12-03-18.bag"
lidar_topic = "/hesai/pandar"
camera_topic = "/alphasense_driver_ros/cam0/color/image/compressed"
save_folder = Path("/home/yifu/tmp/pcd")

# load_lidar_camera_pairs_from_bag(rosbag_path, lidar_topic, camera_topic, save_folder)


config_yaml_path = "/home/yifu/workspace/nerf_data_pipeline/config/fnt_14_ori.yaml"
depth_pose_format = "vilens_slam"
T_cam0_lidar = np.array(
    [
        [-0.99996923, -0.00590096, -0.00516824, -0.04825776],
        [0.00516083, 0.00127056, -0.99998588, -0.06488323],
        [0.00590744, -0.99998178, -0.00124007, -0.08088225],
        [0, 0, 0, 1],
    ]
)
saved_images = list(Path(save_folder).glob("*.jpg"))
saved_npys = list(Path(save_folder).glob("*.npy"))
Path(save_folder / "overlay").mkdir(exist_ok=True)
run_overlay(saved_npys, saved_images, save_folder, config_yaml_path)
