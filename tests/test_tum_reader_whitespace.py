import numpy as np

from oxspires_tools.trajectory.file_interfaces.tum import TUMTrajReader


def test_custom_tum_reader_accepts_blank_lines_and_mixed_whitespace(tmp_path):
    path = tmp_path / "trajectory.txt"
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "\n"
        "1710000000.000000000   1  2  3   0 0 0 1\n"
        "\t1710000000.050000000\t4\t5\t6\t0\t0\t0\t1\t\n"
    )

    trajectory = TUMTrajReader(path, tum_reader_type="custom").read_file()

    assert trajectory.num_poses == 2
    np.testing.assert_allclose(trajectory.positions_xyz, [[1, 2, 3], [4, 5, 6]])
    assert str(trajectory.timestamps[0]) == "1710000000.0"
    assert str(trajectory.timestamps[1]) == "1710000000.05"


def test_custom_tum_reader_handles_indented_comments(tmp_path):
    path = tmp_path / "trajectory.txt"
    path.write_text(
        "   # comment with leading whitespace\n"
        "1710000000.000000000 0 0 0 0 0 0 1\n"
    )

    trajectory = TUMTrajReader(path, tum_reader_type="custom").read_file()

    assert trajectory.num_poses == 1
