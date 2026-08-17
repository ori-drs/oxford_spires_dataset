import cv2
import numpy as np
import pytest

from oxspires_tools.depth.surface_normal import compute_normalmap


def test_fisheye_normal_orientation_uses_undistorted_camera_ray(monkeypatch):
    K = np.eye(3, dtype=np.float64)
    D = np.zeros(4, dtype=np.float64)
    normals = np.array([[1.0, 0.0, -0.1]], dtype=np.float32)
    u = np.array([2])
    v = np.array([0])

    # The pinhole approximation at pixel (2, 0) would use ray [2, 0, 1]
    # and flip this normal. Return a calibrated fisheye ray in the opposite
    # x direction so the expected orientation remains unchanged.
    def fake_undistort_points(points, camera_matrix, distortion, P=None):
        return np.array([[[-2.0, 0.0]]], dtype=np.float64)

    monkeypatch.setattr(cv2.fisheye, "undistortPoints", fake_undistort_points)

    normalmap = compute_normalmap(normals, v, u, h=1, w=3, K=K, D=D, camera_model="OPENCV_FISHEYE")

    expected = ((normals[0] + 1.0) / 2.0 * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(normalmap[0, 2], expected)


def test_unknown_camera_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown camera model"):
        compute_normalmap(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.array([0]),
            np.array([0]),
            h=1,
            w=1,
            K=np.eye(3),
            D=np.zeros(4),
            camera_model="UNKNOWN",
        )
