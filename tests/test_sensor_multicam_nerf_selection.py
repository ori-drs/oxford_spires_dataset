import json

import numpy as np
import pytest

from oxspires_tools.sensor import Sensor


def make_sensor():
    sensor = Sensor.__new__(Sensor)
    sensor.camera_model = "OPENCV_FISHEYE"
    sensor.camera_fov = 160.0
    sensor.camera_topics_labelled = {
        "cam_left": "alphasense_driver_ros_cam1_debayered_image_compressed",
        "cam_right": "alphasense_driver_ros_cam2_debayered_image_compressed",
    }
    return sensor


def test_multicam_nerf_selects_frame_by_camera_parent_folder(tmp_path):
    sensor = make_sensor()
    selected = {}

    def capture_frame(frame):
        selected["file_path"] = frame["file_path"]
        return np.eye(3), np.zeros(4), 1080, 1440

    sensor.get_K_D_h_w_from_colmap_frame = capture_frame
    transforms = {
        "camera_model": "OPENCV_FISHEYE",
        "frames": [
            {"file_path": "images/alphasense_driver_ros_cam1_debayered_image_compressed/1.jpg"},
            {"file_path": "dataset/images/alphasense_driver_ros_cam2_debayered_image_compressed/2.jpg"},
        ],
    }
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(transforms))

    sensor.get_params_for_depth("cam_right", "nerf", path)

    assert selected["file_path"].endswith("alphasense_driver_ros_cam2_debayered_image_compressed/2.jpg")


def test_multicam_nerf_reports_missing_camera_frame(tmp_path):
    sensor = make_sensor()
    transforms = {
        "camera_model": "OPENCV_FISHEYE",
        "frames": [
            {"file_path": "images/alphasense_driver_ros_cam1_debayered_image_compressed/1.jpg"},
        ],
    }
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(transforms))

    with pytest.raises(ValueError, match="No NeRF frame found for camera cam_right"):
        sensor.get_params_for_depth("cam_right", "nerf", path)
