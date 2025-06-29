import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from evo.core.trajectory import xyz_quat_wxyz_to_se3_poses
from pytransform3d.transform_manager import TransformManager

supported_camera_models = ["OPENCV", "OPENCV_FISHEYE", "PINHOLE"]
expected_num_extra_param = {
    "OPENCV": 4,  # k1, k2, p1, p2
    "OPENCV_FISHEYE": 4,  # k1, k2, k3, k4
    "PINHOLE": 0,
}


def get_transformation_matrix(T_AB_t_xyz_q_xyzw):
    assert len(T_AB_t_xyz_q_xyzw) == 7, f"only got {len(T_AB_t_xyz_q_xyzw)} params"
    t_xyz = T_AB_t_xyz_q_xyzw[:3]
    q_xyzw = T_AB_t_xyz_q_xyzw[3:]
    q_wxyz = [q_xyzw[-1]] + q_xyzw[:-1]
    return xyz_quat_wxyz_to_se3_poses([t_xyz], [q_wxyz])[0]


@dataclass
class Camera:
    label: str
    topic: str
    image_height: int
    image_width: int
    intrinsics: List[float]  # fx, fy, cx, cy
    extra_params: List[float]
    T_cam_lidar_t_xyz_q_xyzw_overwrite: List[float] = field(
        default_factory=list
    )  # overwrite that computed from T_cam_imu and T_base_lidar
    T_cam_imu_t_xyz_q_xyzw: List[float] = field(default_factory=list)
    camera_model: str = "OPENCV_FISHEYE"  # https://colmap.github.io/cameras.html

    def __post_init__(self):
        assert self.camera_model in supported_camera_models, f"Camera model {self.camera_model} not supported"
        assert len(self.intrinsics) == 4, f"Expected 4 intrinsics, got {len(self.intrinsics)}"
        assert len(self.extra_params) == expected_num_extra_param[self.camera_model], (
            f"Expected {expected_num_extra_param[self.camera_model]} extra params, got {len(self.extra_params)}"
        )
        self.T_cam_lidar_overwrite = (
            get_transformation_matrix(self.T_cam_lidar_t_xyz_q_xyzw_overwrite)
            if len(self.T_cam_lidar_t_xyz_q_xyzw_overwrite) > 0
            else None
        )
        self.T_cam_imu = get_transformation_matrix(self.T_cam_imu_t_xyz_q_xyzw)

    def get_K(self):
        fx, fy, cx, cy = self.intrinsics
        return np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )


@dataclass
class Sensor:
    cameras: List[Camera] = field(default_factory=list)
    # lidars: List[Camera] = field(default_factory=list)
    max_time_diff_camera_and_pose: float = 0.025
    camera_model: str = "OPENCV_FISHEYE"  # https://colmap.github.io/cameras.html
    T_base_imu_t_xyz_q_xyzw: List[float] = field(default_factory=list)
    T_base_lidar_t_xyz_q_xyzw: List[float] = field(default_factory=list)
    tf: TransformManager = field(init=False)

    @staticmethod
    def convert_camera_topic_to_folder_name(camera_topic):
        camera_topic = camera_topic.replace("/", "_")
        if camera_topic.startswith("_"):
            camera_topic = camera_topic[1:]
        return camera_topic

    def set_sensor_frames(self):
        self.tf = TransformManager()
        self.tf.add_transform("imu", "base", self.T_base_imu)
        self.tf.add_transform("base", "lidar", self.T_base_lidar)
        for camera in self.cameras:
            self.tf.add_transform("imu", camera.label, camera.T_cam_imu)

    def __post_init__(self):
        self.cameras = [Camera(**camera) for camera in self.cameras]
        for idx, camera in enumerate(self.cameras):
            camera.idx = idx
        self.camera_topics_raw = [camera.topic for camera in self.cameras]
        self.camera_topics_labelled = {
            camera.label: Sensor.convert_camera_topic_to_folder_name(camera.topic) for camera in self.cameras
        }
        self.camera_topics_labelled_reverse = {v: k for k, v in self.camera_topics_labelled.items()}
        self.camera_topics = list(self.camera_topics_labelled.values())
        self.camera_subdirs = self.camera_topics

        self.cam_idx_new = {camera.label: (camera.idx) for camera in self.cameras}
        for cam_name in self.camera_topics_labelled.keys():
            # self.cam_ids[self.camera_topics_labelled[cam_name]] = self.cam_ids[cam_name]
            self.cam_idx_new[self.camera_topics_labelled[cam_name]] = self.cam_idx_new[cam_name]
        self.T_base_imu = get_transformation_matrix(self.T_base_imu_t_xyz_q_xyzw)
        self.T_base_lidar = get_transformation_matrix(self.T_base_lidar_t_xyz_q_xyzw)
        self.set_sensor_frames()

        self.T_cam_base_overwrite = {
            camera.label: camera.T_cam_lidar_overwrite @ np.linalg.inv(self.T_base_lidar)
            for camera in self.cameras
            if camera.T_cam_lidar_overwrite is not None
        }

    def get_colmap_cam_id(self, camera_name=None, camera_topic=None):
        # colmap cam id starts from 1, not 0
        if camera_name is not None and camera_topic is None:
            return self.cam_idx_new[camera_name] + 1
        elif camera_name is None and camera_topic is not None:
            return self.cam_idx_new[self.camera_topics_labelled_reverse[camera_topic]] + 1
        else:
            raise ValueError("Only one of camera_name or camera_topic should be specified")

    def get_camera(self, camera_name):
        return self.cameras[self.cam_idx_new[camera_name]]

    def get_params_for_depth(self, cam_name, colmap_json_path: Path = None):
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

        fov_deg = 140.0  # TODO!
        if colmap_json_path is None:
            print("Depth Image: Using Frontier_config / Kalibr for Intrinsics")
            K = self.get_camera(cam_name).get_K()
            D = np.array(self.get_camera(cam_name).extra_params)
            h = self.get_camera(cam_name).image_height
            w = self.get_camera(cam_name).image_width
            print(f"{cam_name} K: {K}, D: {D}, h: {h}, w: {w}")
        else:
            print("Depth Image: Using NeRF transforms.json 's Intrinsics (from colmap)")
            colmap_traj = json.load(open(colmap_json_path, "r"))
            if len(self.camera_topics_labelled) > 1:
                assert list(colmap_traj.keys()) == ["camera_model", "frames"]
                assert colmap_traj["camera_model"] == self.camera_model
                for frame in colmap_traj["frames"]:
                    if frame["file_path"].split("/")[1] == self.camera_topics_labelled[cam_name]:
                        K, D, h, w = get_K_D_h_w_from_colmap_frame(frame, self.camera_model, cam_name)
                        break
            elif len(self.camera_topics_labelled) == 1:
                assert colmap_traj["camera_model"] == self.camera_model
                K, D, h, w = get_K_D_h_w_from_colmap_frame(colmap_traj, self.camera_model, cam_name)
            else:
                raise RuntimeError("Invalid camera_topics_labelled")

        return K, D, h, w, fov_deg, self.camera_model
