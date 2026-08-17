import numpy as np
import open3d as o3d

from oxspires_tools.eval import save_error_cloud


def test_zero_error_cloud_has_finite_colours(monkeypatch, tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distances = np.zeros(2)
    captured = {}

    def capture_write(path, cloud):
        captured["points"] = np.asarray(cloud.points).copy()
        captured["colors"] = np.asarray(cloud.colors).copy()
        return True

    monkeypatch.setattr(o3d.io, "write_point_cloud", capture_write)

    save_error_cloud(points, tmp_path / "perfect.ply", distances=distances)

    np.testing.assert_allclose(captured["points"], points)
    assert np.isfinite(captured["colors"]).all()
    assert captured["colors"].shape == (2, 3)


def test_error_cloud_returns_when_distance_filter_removes_all_points(monkeypatch, tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distances = np.array([1.0, 2.0])
    writes = []

    monkeypatch.setattr(o3d.io, "write_point_cloud", lambda *args, **kwargs: writes.append(args))

    save_error_cloud(points, tmp_path / "empty.ply", distances=distances, max_distance=0.5)

    assert writes == []
