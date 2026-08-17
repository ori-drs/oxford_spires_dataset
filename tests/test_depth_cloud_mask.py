import numpy as np

import oxspires_tools.depth.main as depth_main


class DummyPointCloud:
    def __init__(self, points, normals=None):
        self.points = np.asarray(points, dtype=float)
        self.normals = np.asarray(normals if normals is not None else [], dtype=float)


def test_cloud_mask_preserves_hidden_point_removal_index_order(monkeypatch):
    source_cloud = DummyPointCloud(
        [
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 2.0],
        ]
    )

    # Emulate HPR returning original indices in a non-sorted order. The
    # visible point cloud has the same order as the returned index map:
    # visible[0] -> source[2], visible[1] -> source[0].
    visible_cloud = DummyPointCloud(
        [
            source_cloud.points[2],
            source_cloud.points[0],
        ]
    )
    monkeypatch.setattr(depth_main, "apply_hidden_point_removal", lambda _: (visible_cloud, [2, 0]))

    # Keep only visible[0] through the FOV and image projection stages.
    monkeypatch.setattr(depth_main, "filter_points_outside_fov", lambda points, fov: np.array([True, False]))
    monkeypatch.setattr(
        depth_main,
        "project_points_on_image",
        lambda points, K, D, w, h, camera_model: (np.array([[1.0, 1.0]]), np.array([True])),
    )
    monkeypatch.setattr(
        depth_main,
        "encode_points_as_depthmap",
        lambda *args, **kwargs: (
            np.zeros((3, 3), dtype=np.uint16),
            np.array([True]),
            np.array([1]),
            np.array([1]),
            np.array([0]),
        ),
    )

    _, _, cloud_mask = depth_main.get_depth_from_cloud(
        source_cloud,
        K=np.eye(3),
        D=np.zeros(4),
        w=3,
        h=3,
        fov_deg=160.0,
        camera_model="OPENCV_FISHEYE",
        compute_cloud_mask=True,
    )

    np.testing.assert_array_equal(cloud_mask, np.array([False, False, True]))
