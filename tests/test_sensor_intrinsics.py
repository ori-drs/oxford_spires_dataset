import numpy as np

from oxspires_tools.sensor import Sensor


def test_colmap_frame_builds_homogeneous_intrinsic_matrix():
    sensor = Sensor.__new__(Sensor)
    sensor.camera_model = "OPENCV_FISHEYE"
    frame = {
        "fl_x": 700.0,
        "fl_y": 710.0,
        "cx": 720.0,
        "cy": 540.0,
        "k1": -0.01,
        "k2": 0.002,
        "k3": 0.0,
        "k4": 0.0,
        "h": 1080,
        "w": 1440,
    }

    K, D, h, w = sensor.get_K_D_h_w_from_colmap_frame(frame)

    np.testing.assert_allclose(
        K,
        np.array(
            [
                [700.0, 0.0, 720.0],
                [0.0, 710.0, 540.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(D, [-0.01, 0.002, 0.0, 0.0])
    assert (h, w) == (1080, 1440)
