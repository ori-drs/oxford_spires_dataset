import numpy as np

from oxspires_tools.depth.projection import encode_points_as_depthmap


def test_depth_encoding_rejects_invalid_values_and_keeps_uint16_max():
    factor = 256.0
    max_depth = np.iinfo(np.uint16).max / factor
    points_on_img = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=float)
    points_in_3d = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, np.nan],
            [0.0, 0.0, np.inf],
            [0.0, 0.0, max_depth],
        ],
        dtype=float,
    )

    depthmap, z_mask, _, _, _ = encode_points_as_depthmap(
        points_on_img,
        points_in_3d,
        h=1,
        w=6,
        is_euclidean=False,
        depth_encode_factor=factor,
    )

    np.testing.assert_array_equal(z_mask, [True, False, False, False, False, True])
    assert depthmap[0, 0] == 256
    assert depthmap[0, 5] == np.iinfo(np.uint16).max
    assert np.count_nonzero(depthmap) == 2
