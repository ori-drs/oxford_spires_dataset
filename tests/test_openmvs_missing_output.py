import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "reconstruction_benchmark" / "mvs.py"


def load_mvs_module():
    spec = importlib.util.spec_from_file_location("oxspires_reconstruction_mvs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_openmvs_fails_if_dense_cloud_was_not_generated(tmp_path, monkeypatch):
    mvs = load_mvs_module()
    commands = []
    read_attempted = False

    monkeypatch.setattr(mvs, "run_command", lambda command, print_command=True: commands.append(command))

    def unexpected_read(*args, **kwargs):
        nonlocal read_attempted
        read_attempted = True
        raise AssertionError("Open3D should not be asked to read a missing dense cloud")

    monkeypatch.setattr(mvs.o3d.io, "read_point_cloud", unexpected_read)

    colmap_dir = tmp_path / "colmap"
    mvs_dir = tmp_path / "mvs"

    with pytest.raises(FileNotFoundError, match="scene_dense.ply"):
        mvs.run_openmvs(
            image_path=tmp_path / "images",
            colmap_output_path=colmap_dir,
            sparse_folder=colmap_dir / "sparse" / "0",
            mvs_dir=mvs_dir,
            openmvs_bin="/opt/OpenMVS",
        )

    assert len(commands) == 2
    assert "InterfaceCOLMAP" in commands[0]
    assert "DensifyPointCloud" in commands[1]
    assert read_attempted is False
