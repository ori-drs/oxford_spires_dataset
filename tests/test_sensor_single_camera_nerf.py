import json

import numpy as np
import pytest

from oxspires_tools.sensor import Sensor


def _make_sensor_stub():
    sensor = Sensor.__new__(Sensor)
    sensor.camera_model = "OPENCV_FISHEYE"
    sensor.camera_fov = 160.0
    sensor.camera_topics_labelled = {"cam0": "cam0"}
    return sensor


def test_single_camera_nerf_uses_first_frame_intrinsics(tmp_path):
    sensor = _make_sensor_stub()
    seen = {}

    def parse_frame(frame):
        seen["frame"] = frame
        return np.eye(3), np.zeros(4), 1080, 1440

    sensor.get_K_D_h_w_from_colmap_frame = parse_frame
    frame = {
        "file_path": "images/cam0/1710000000.000000000.jpg",
        "fl_x": 700.0,
        "fl_y": 700.0,
        "cx": 720.0,
        "cy": 540.0,
    }
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"camera_model": "OPENCV_FISHEYE", "frames": [frame]}))

    K, D, h, w, fov, model = sensor.get_params_for_depth("cam0", "nerf", path)

    assert seen["frame"] == frame
    np.testing.assert_array_equal(K, np.eye(3))
    np.testing.assert_array_equal(D, np.zeros(4))
    assert (h, w, fov, model) == (1080, 1440, 160.0, "OPENCV_FISHEYE")


def test_single_camera_nerf_rejects_empty_frame_list(tmp_path):
    sensor = _make_sensor_stub()
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"camera_model": "OPENCV_FISHEYE", "frames": []}))

    with pytest.raises(ValueError, match="contains no frames"):
        sensor.get_params_for_depth("cam0", "nerf", path)
