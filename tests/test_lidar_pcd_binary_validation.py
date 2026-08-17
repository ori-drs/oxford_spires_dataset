import struct

import numpy as np
import pytest

from oxspires_tools.lidar_undistortion.io import read_pcd_binary


HEADER = """VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 1
HEIGHT 1
POINTS 1
"""


def test_read_pcd_binary_rejects_missing_data_header(tmp_path):
    path = tmp_path / "missing_data.pcd"
    path.write_text(HEADER)

    with pytest.raises(ValueError, match="before a DATA field"):
        read_pcd_binary(path)


def test_read_pcd_binary_rejects_ascii_payload(tmp_path):
    path = tmp_path / "ascii.pcd"
    path.write_text(HEADER + "DATA ascii\n1 2 3\n")

    with pytest.raises(ValueError, match="Expected DATA binary"):
        read_pcd_binary(path)


def test_read_pcd_binary_rejects_truncated_payload(tmp_path):
    path = tmp_path / "truncated.pcd"
    path.write_bytes((HEADER + "DATA binary\n").encode("ascii") + struct.pack("<ff", 1.0, 2.0))

    with pytest.raises(ValueError, match="Truncated binary PCD payload"):
        read_pcd_binary(path)


def test_read_pcd_binary_reads_complete_payload(tmp_path):
    path = tmp_path / "complete.pcd"
    path.write_bytes((HEADER + "DATA binary\n").encode("ascii") + struct.pack("<fff", 1.0, 2.0, 3.0))

    cloud, header = read_pcd_binary(path)

    assert header["DATA"] == "binary"
    assert cloud.shape == (1,)
    np.testing.assert_allclose([cloud["x"][0], cloud["y"][0], cloud["z"][0]], [1.0, 2.0, 3.0])
