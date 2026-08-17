import pytest

from oxspires_tools.point_cloud import read_pcd_with_viewpoint


PCD_HEADER_PREFIX = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 1
HEIGHT 1
POINTS 1
"""


def test_read_pcd_with_viewpoint_rejects_missing_viewpoint(tmp_path):
    path = tmp_path / "missing_viewpoint.pcd"
    path.write_text(PCD_HEADER_PREFIX + "DATA ascii\n0 0 0\n")

    with pytest.raises(ValueError, match="no VIEWPOINT header"):
        read_pcd_with_viewpoint(str(path))


def test_read_pcd_with_viewpoint_rejects_malformed_viewpoint(tmp_path):
    path = tmp_path / "malformed_viewpoint.pcd"
    path.write_text(PCD_HEADER_PREFIX + "VIEWPOINT 0 0 0 1 0 0\nDATA ascii\n0 0 0\n")

    with pytest.raises(ValueError, match="Invalid VIEWPOINT header"):
        read_pcd_with_viewpoint(str(path))
