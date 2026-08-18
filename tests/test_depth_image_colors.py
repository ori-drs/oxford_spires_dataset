import cv2
import numpy as np

from oxspires_tools.depth.projection import decode_points_from_depthmap


def test_decode_depth_normalizes_uint8_image_colors(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "undistortPoints",
        lambda points, K, D, P=None: np.array([[[0.0, 0.0]]], dtype=np.float64),
    )

    depth = np.array([[256]], dtype=np.uint16)
    image = np.array([[[255, 128, 0]]], dtype=np.uint8)

    pcd = decode_points_from_depthmap(
        depth,
        K=np.eye(3),
        D=np.zeros(4),
        is_euclidean=False,
        depth_encode_factor=256.0,
        camera_model="OPENCV",
        image=image,
    )

    np.testing.assert_allclose(np.asarray(pcd.points), [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.asarray(pcd.colors), [[1.0, 128.0 / 255.0, 0.0]])


def test_decode_depth_preserves_normalized_float_colors(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "undistortPoints",
        lambda points, K, D, P=None: np.array([[[0.0, 0.0]]], dtype=np.float64),
    )

    depth = np.array([[256]], dtype=np.uint16)
    image = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float64)

    pcd = decode_points_from_depthmap(
        depth,
        K=np.eye(3),
        D=np.zeros(4),
        is_euclidean=False,
        depth_encode_factor=256.0,
        camera_model="OPENCV",
        image=image,
    )

    np.testing.assert_allclose(np.asarray(pcd.colors), image.reshape(1, 3))
