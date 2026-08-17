import numpy as np
import pytest

from oxspires_tools.depth.projection import decode_points_from_depthmap


@pytest.mark.parametrize("camera_model", ["OPENCV_FISHEYE", "OPENCV"])
def test_decode_empty_depthmap_returns_empty_cloud(camera_model):
    depth = np.zeros((4, 5), dtype=np.uint16)
    image = np.zeros((4, 5, 3), dtype=np.float32)
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])
    D = np.zeros(4)

    cloud = decode_points_from_depthmap(
        depth,
        K,
        D,
        is_euclidean=False,
        depth_encode_factor=256.0,
        camera_model=camera_model,
        image=image,
    )

    assert len(cloud.points) == 0
    assert len(cloud.colors) == 0
