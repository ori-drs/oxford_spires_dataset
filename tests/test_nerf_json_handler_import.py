import json

from oxspires_tools.trajectory.nerf_json_handler import NeRFJsonHandler


def test_nerf_json_handler_imports_and_loads_as_package(tmp_path):
    json_path = tmp_path / "transforms.json"
    json_path.write_text(json.dumps({"frames": []}))

    handler = NeRFJsonHandler(json_path)

    assert handler.get_n_frames() == 0
