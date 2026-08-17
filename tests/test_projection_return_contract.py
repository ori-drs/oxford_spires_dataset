import cv2
import numpy as np

from oxspires_tools.depth.projection import project_points_on_image


def test_project_points_none_result_returns_two_values(monkeypatch):
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    D = np.zeros(4, dtype=np.float64)

    def return_none(*args, **kwargs):
        return None, None

    monkeypatch.setattr(cv2.fisheye, "projectPoints", return_none)

    projected, valid_mask = project_points_on_image(points, K, D, 640, 480, "OPENCV_FISHEYE")

    assert projected.shape == (0, 2)
    assert valid_mask.shape == (2,)
    assert valid_mask.dtype == bool
    assert not valid_mask.any()
