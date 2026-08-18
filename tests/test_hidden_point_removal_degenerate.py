import numpy as np
import open3d as o3d

from oxspires_tools.depth.utils import apply_hidden_point_removal


def test_hidden_point_removal_accepts_empty_cloud():
    cloud = o3d.geometry.PointCloud()

    visible, indices = apply_hidden_point_removal(cloud)

    assert len(visible.points) == 0
    assert indices == []


def test_hidden_point_removal_preserves_coincident_points():
    cloud = o3d.geometry.PointCloud()
    points = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    cloud.points = o3d.utility.Vector3dVector(points)

    visible, indices = apply_hidden_point_removal(cloud)

    assert indices == [0, 1]
    np.testing.assert_allclose(np.asarray(visible.points), points)
