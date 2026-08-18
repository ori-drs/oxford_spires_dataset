import numpy as np

from oxspires_tools.sensor import Sensor


class DummyCamera:
    camera_model = "PINHOLE"
    extra_params = []
    image_height = 480
    image_width = 640

    def get_K(self):
        return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_vilens_depth_uses_selected_camera_model_not_sensor_default():
    sensor = Sensor.__new__(Sensor)
    sensor.cameras = [DummyCamera()]
    sensor.cam_idx_new = {"cam0": 0}
    sensor.camera_model = "OPENCV_FISHEYE"
    sensor.camera_fov = 120.0

    K, D, h, w, fov, camera_model = sensor.get_params_for_depth("cam0", "vilens_slam")

    np.testing.assert_allclose(
        K,
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
    )
    assert D.size == 0
    assert (h, w) == (480, 640)
    assert fov == 120.0
    assert camera_model == "PINHOLE"
