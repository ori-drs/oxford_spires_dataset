import numpy as np
from evo.core.trajectory import PoseTrajectory3D

from oxspires_tools.trajectory.file_interfaces.vilens_slam import VilensSlamTrajReader, VilensSlamTrajWriter


def test_vilens_writer_preserves_seconds_and_nanoseconds(tmp_path):
    timestamps = np.array([np.float128("1710000000.000000123"), np.float128("1710000000.000000456")])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    quaternions_wxyz = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    trajectory = PoseTrajectory3D(positions, quaternions_wxyz, timestamps=timestamps)

    output = tmp_path / "slam-poses.csv"
    VilensSlamTrajWriter(str(output)).write_file(trajectory)

    rows = [line for line in output.read_text().splitlines() if not line.startswith("#")]
    assert rows[0].split(", ")[:3] == ["0", "1710000000", "000000123"]
    assert rows[1].split(", ")[:3] == ["1", "1710000000", "000000456"]

    loaded = VilensSlamTrajReader(str(output)).read_file()
    np.testing.assert_array_equal(loaded.timestamps, timestamps)
    np.testing.assert_allclose(loaded.positions_xyz, positions)
    np.testing.assert_allclose(loaded.orientations_quat_wxyz, quaternions_wxyz)
