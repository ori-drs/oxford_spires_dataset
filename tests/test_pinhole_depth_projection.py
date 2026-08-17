import numpy as np

from oxspires_tools.depth.projection import decode_points_from_depthmap, project_points_on_image


def _pinhole_calibration():
    K = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return K, np.array([])


def test_pinhole_projection_uses_intrinsic_matrix_without_distortion():
    K, D = _pinhole_calibration()
    points = np.array([[0.0, 0.0, 2.0], [0.2, -0.1, 2.0]])

    pixels, mask = project_points_on_image(points, K, D, w=100, h=80, camera_model="PINHOLE")

    np.testing.assert_array_equal(mask, [True, True])
    np.testing.assert_array_equal(pixels, [[50.0, 40.0], [60.0, 35.0]])


def test_pinhole_depth_decode_recovers_camera_frame_point():
    K, D = _pinhole_calibration()
    depth = np.zeros((80, 100), dtype=np.uint16)
    depth[40, 50] = 512  # 2 metres at factor 256

    cloud = decode_points_from_depthmap(
        depth,
        K,
        D,
        is_euclidean=False,
        depth_encode_factor=256.0,
        camera_model="PINHOLE",
    )

    np.testing.assert_allclose(np.asarray(cloud.points), [[0.0, 0.0, 2.0]], atol=1e-12)
