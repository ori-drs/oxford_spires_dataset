import importlib.util
from pathlib import Path

import numpy as np


def load_generate_depth_module():
    script_path = Path(__file__).parents[1] / "scripts" / "generate_depth.py"
    spec = importlib.util.spec_from_file_location("generate_depth_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyPointCloud:
    def __init__(self):
        self.points = np.zeros((1, 3))
        self.transforms = []

    def transform(self, transform):
        self.transforms.append(transform)


def test_process_cloud_forwards_skip_hpr(monkeypatch, tmp_path):
    module = load_generate_depth_module()
    point_cloud = DummyPointCloud()
    captured = {}

    monkeypatch.setattr(module.o3d.io, "read_point_cloud", lambda _: point_cloud)

    def fake_get_depth_from_cloud(
        pcd,
        K,
        D,
        w,
        h,
        fov_deg,
        camera_model,
        depth_factor=256.0,
        is_euclidean=False,
        compute_cloud_mask=False,
        skip_hpr=False,
    ):
        captured["depth_factor"] = depth_factor
        captured["is_euclidean"] = is_euclidean
        captured["compute_cloud_mask"] = compute_cloud_mask
        captured["skip_hpr"] = skip_hpr
        return np.zeros((2, 2), dtype=np.uint16), None

    monkeypatch.setattr(module, "get_depth_from_cloud", fake_get_depth_from_cloud)
    monkeypatch.setattr(module, "save_projection_outputs", lambda *args, **kwargs: None)

    module.process_cloud_to_depth(
        (tmp_path / "123.jpg", tmp_path / "123.pcd", 0.0),
        T_cam_base=np.eye(4),
        K=np.eye(3),
        D=np.zeros(4),
        w=2,
        h=2,
        fov_deg=180.0,
        camera_model="OPENCV_FISHEYE",
        depth_factor=512.0,
        euclidean=True,
        skip_hpr=True,
        output_depth_dir=tmp_path / "depth",
        output_normal_dir=tmp_path / "normal",
        overlay_dir=tmp_path / "overlay",
        cam_dir_name="cam0",
    )

    assert captured == {
        "depth_factor": 512.0,
        "is_euclidean": True,
        "compute_cloud_mask": False,
        "skip_hpr": True,
    }
