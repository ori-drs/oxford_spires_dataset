import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "reconstruction_benchmark" / "nerf_json.py"


class JsonHandlerStub:
    def __init__(self, input_json_path):
        self.traj = json.loads(Path(input_json_path).read_text())

    def sort_frames(self):
        self.traj["frames"].sort(key=lambda frame: frame["file_path"])

    def save_json(self, output_path):
        Path(output_path).write_text(json.dumps(self.traj))


def load_split_function():
    tree = ast.parse(SCRIPT_PATH.read_text())
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_json_every_n"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"NeRFJsonHandler": JsonHandlerStub, "deepcopy": deepcopy}
    exec(compile(module, SCRIPT_PATH.as_posix(), "exec"), namespace)
    return namespace["split_json_every_n"]


def test_split_json_every_n_uses_requested_stride_and_complement(tmp_path):
    split_json_every_n = load_split_function()
    input_path = tmp_path / "transforms.json"
    kept_path = tmp_path / "kept.json"
    removed_path = tmp_path / "removed.json"

    frames = [{"file_path": f"images/{index}.jpg", "id": index} for index in [6, 2, 5, 0, 4, 1, 3]]
    input_path.write_text(json.dumps({"camera_model": "OPENCV_FISHEYE", "frames": frames}))

    split_json_every_n(input_path, 3, removed_path, kept_path)

    kept = json.loads(kept_path.read_text())
    removed = json.loads(removed_path.read_text())

    assert [frame["id"] for frame in kept["frames"]] == [0, 3, 6]
    assert [frame["id"] for frame in removed["frames"]] == [1, 2, 4, 5]
    assert kept["camera_model"] == "OPENCV_FISHEYE"
    assert removed["camera_model"] == "OPENCV_FISHEYE"
    assert {frame["id"] for frame in kept["frames"]}.isdisjoint(
        {frame["id"] for frame in removed["frames"]}
    )


def test_split_json_every_n_rejects_invalid_stride(tmp_path):
    split_json_every_n = load_split_function()

    with pytest.raises(ValueError, match="positive integer"):
        split_json_every_n(tmp_path / "unused.json", 0, tmp_path / "removed.json", tmp_path / "kept.json")
