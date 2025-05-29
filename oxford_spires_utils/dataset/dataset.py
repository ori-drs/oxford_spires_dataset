import re
from pathlib import Path

import numpy as np
import open3d as o3d
import pyrender
import trimesh
import yaml
from evo.core.trajectory import PosePath3D

from oxford_spires_utils.dataset.sensor import Sensor
from oxford_spires_utils.dataset.utils import find_closest_in_sorted
from oxford_spires_utils.pose.file_interfaces import VilensSlamTrajReader
from oxford_spires_utils.pose.file_interfaces.timestamp import TimeStamp
from oxford_spires_utils.pose.json_handler import JsonHandler


class VilensSlamOutputHandler:
    def __init__(self, traj_path: Path, slam_clouds_folder_path: Path, downsample_to: int = -1):
        self.traj_path = Path(traj_path)
        self.slam_clouds_folder_path = Path(slam_clouds_folder_path)
        self.traj = self.get_traj(self.traj_path)
        self.check_clouds_with_poses()
        if downsample_to > 0:
            print(f"Downsampling to {downsample_to} poses")
            self.traj.downsample(downsample_to)
        self.num_poses = self.traj.num_poses

    def get_traj(self, traj_path: Path):
        return VilensSlamTrajReader(traj_path).read_file()

    def get_cloud_path(self, pose_idx: int):
        timestamp = TimeStamp(t_float128=self.traj.timestamps[pose_idx])
        timestamp_str = timestamp.t_string
        timestamp_str = timestamp_str.replace(".", "_")  # vilens slam output convention
        cloud_file = self.slam_clouds_folder_path / f"cloud_{timestamp_str}.pcd"
        return cloud_file

    def get_cloud_pcd(self, pose_idx: int):
        cloud_file = self.get_cloud_path(pose_idx)
        return o3d.io.read_point_cloud(str(cloud_file))

    def get_pose(self, pose_idx: int):
        return self.traj.poses_se3[pose_idx]

    def check_clouds_with_poses(self):
        cloud_files = list(self.slam_clouds_folder_path.glob("*.pcd"))
        assert all([self.get_cloud_path(pose_idx) in cloud_files for pose_idx in range(self.traj.num_poses)])


class FrontierDataset:
    def __init__(
        self,
        poses_path: Path,
        slam_clouds_folder_path: Path,
        sensor_config_yaml_path: Path,
        transform_json_path: Path = None,
        image_folder_path: Path = None,
        slam_pose_downsample_to: int = -1,  # downsample poses. -1 means no downsampling
        synced_image_step: int = 1,  # only load synced images at step intervals
        image_ext: str = ".jpg",
    ):
        self.vilens_slam_handler = VilensSlamOutputHandler(poses_path, slam_clouds_folder_path, slam_pose_downsample_to)
        with open(sensor_config_yaml_path, "r") as f:
            self.sensor = Sensor(**yaml.safe_load(f)["sensor"])
        if image_folder_path is not None:
            self.image_folder_path = Path(image_folder_path)
            self.load_synced_images(image_ext, synced_image_step)
        if transform_json_path is not None:
            self.json_handler = JsonHandler(transform_json_path)

    def get_pose(self, pose_idx: int, sensor_frame="base", source="slam"):
        """
        sensor_frame: "base", "cam_left", "cam_front", "cam_right"
        """
        if source == "slam":
            T_world_base = self.vilens_slam_handler.get_pose(pose_idx)
            if sensor_frame == "base":
                return T_world_base
            elif sensor_frame in [camera.label for camera in self.sensor.cameras]:
                T_cam_base = (
                    self.sensor.T_cam_base_overwrite[sensor_frame]
                    if sensor_frame in self.sensor.T_cam_base_overwrite
                    else np.linalg.inv(
                        self.sensor.tf.get_transform(sensor_frame, "base")
                    )  # T_base_cam = self.tf.get_transform(cam_name, "base")
                )
                T_world_base = self.vilens_slam_handler.traj.poses_se3[pose_idx]
                T_world_cam = T_world_base @ np.linalg.inv(T_cam_base)
                return T_world_cam
            else:
                raise ValueError(f"Unknown sensor frame {sensor_frame}")
        elif source == "json":
            raise NotImplementedError("Json source not implemented")

    def get_all_camera_poses(self, return_dict=False, sync_with_images=False):
        """
        Each pose has three cameras. This function returns all camera poses in a list or dict
        if sync_with_images is True, only return poses which have synced images. Returned
        Returns:
            PosePath3D or dict: if return_dict is True, returns dict with camera label as key
        """
        camera_poses = {} if return_dict else []
        camera_paths = {} if return_dict else []
        for camera in self.sensor.cameras:
            if return_dict:
                camera_poses[camera.label] = []
                camera_paths[camera.label] = []
            T_cam_base = (
                self.sensor.T_cam_base_overwrite[camera.label]
                if camera.label in self.sensor.T_cam_base_overwrite
                else self.sensor.tf.get_transform("base", camera.label)
            )
            for i in range(self.vilens_slam_handler.num_poses):
                if sync_with_images and self.image_paths[camera.label][i] is None:
                    print(f"Cannot find {camera.label} pose {i} at {self.vilens_slam_handler.traj.timestamps[i]}")
                    continue
                T_world_base = self.vilens_slam_handler.traj.poses_se3[i]
                T_world_cam = T_world_base @ np.linalg.inv(T_cam_base)
                if return_dict:
                    camera_poses[camera.label].append(T_world_cam)
                    camera_paths[camera.label].append(self.image_paths[camera.label][i])
                else:
                    camera_poses.append(T_world_cam)
                    camera_paths.append(self.image_paths[camera.label][i])
        if not return_dict:
            camera_poses = np.array(camera_poses)
            camera_poses = PosePath3D(poses_se3=camera_poses)
        return camera_poses, camera_paths

    def load_synced_images(self, image_ext=".jpg", step=1):
        """ "
        Load synced images with slam poses.
        If the time difference between image and pose is larger than max_time_diff_camera_and_pose,
        the image_path will be None
        If step is larger than 1, only load images at step intervals. Skipped image paths will be None
        Return: dict with camera label as key and list of image paths as value
        """
        self.image_paths = {}
        for camera in self.sensor.cameras:
            image_timestamps = []
            image_paths = {}
            cam_topic_folder = self.sensor.camera_topics_labelled[camera.label]
            image_folder_path = self.image_folder_path / cam_topic_folder
            for it in sorted(list(image_folder_path.glob(f"*{image_ext}"))):
                ret = re.findall(r"\d+", it.name)
                timestamp = float(".".join(ret))
                image_timestamps.append(timestamp)
                image_paths[timestamp] = it
            assert len(image_paths) > 0, "No images are found"

            # Collect all image and lidar pair which have timestamp close enough
            synced_image_list = []
            for i in range(self.vilens_slam_handler.num_poses):
                image_timestamp, diff, _ = find_closest_in_sorted(
                    image_timestamps, self.vilens_slam_handler.traj.timestamps[i]
                )
                synced_image_path = (
                    image_paths[image_timestamp]
                    if (diff < self.sensor.max_time_diff_camera_and_pose and i % step == 0)
                    else None
                )
                synced_image_list.append(synced_image_path)
            real_synced_image_len = len([x for x in synced_image_list if x is not None])
            if real_synced_image_len < self.vilens_slam_handler.num_poses:
                print(
                    f"Only synced {real_synced_image_len}/{self.vilens_slam_handler.num_poses} poses in {cam_topic_folder}"
                )
            else:
                print(f"All {self.vilens_slam_handler.num_poses} poses are synced with images in {cam_topic_folder}")
            self.image_paths[camera.label] = synced_image_list

    def get_map_cloud(self, step=1, downsample_res=0.1, save_path=None):
        map_cloud = o3d.geometry.PointCloud()
        for i in range(0, self.vilens_slam_handler.num_poses, step):
            # read cloud
            cloud = self.vilens_slam_handler.get_cloud_pcd(i)
            T_WB = self.vilens_slam_handler.get_pose(i)
            cloud.transform(T_WB)
            map_cloud += cloud
        map_cloud = map_cloud.voxel_down_sample(downsample_res)
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, map_cloud)
        return map_cloud

    def get_map_mesh(self, mesh_path):
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles))
        mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
        return mesh
