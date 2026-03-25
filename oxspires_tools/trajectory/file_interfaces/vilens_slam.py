import logging
import os

import evo
import numpy as np
from evo.tools.file_interface import csv_read_matrix

from .base import BasicTrajReader, BasicTrajWriter
from .timestamp import TimeStamp

logger = logging.getLogger(__name__)


class VilensSlamTrajReader(BasicTrajReader):
    """Read VILENS SLAM trajectory file."""

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path)

    def read_file(self):
        """Read VILENS SLAM trajectory file."""
        raw_mat = csv_read_matrix(self.file_path, delim=",", comment_str="#")
        if not raw_mat:
            logger.error(f"Empty or unreadable VILENS SLAM file: {self.file_path}")
            raise ValueError()
        timestamps = [TimeStamp(sec=row[1], nsec=row[2]).t_float128 for row in raw_mat]
        mat = np.array(raw_mat).astype(float)
        xyz = mat[:, 3:6]
        quat_xyzw = mat[:, 6:10]
        quat_wxyz = np.roll(quat_xyzw, 1, axis=1)  # xyzw -> wxyz
        return evo.core.trajectory.PoseTrajectory3D(xyz, quat_wxyz, timestamps=timestamps)


class VilensSlamTrajWriter(BasicTrajWriter):
    """Write VILENS SLAM trajectory format file (poses.csv)."""

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path)

    def write_file(self, pose):
        """Write trajectory in CSV style VILENS SLAM format."""

        if not isinstance(pose, evo.core.trajectory.PoseTrajectory3D):
            logger.error(f"pose should be PoseTrajectory3D from evo, got: {type(pose)}")
            raise ValueError()
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))

        assert pose.check()[0] is True, pose.check()[1]

        logger.info(f"Writing pose message in CSV format to: {self.file_path}")
        with open(self.file_path, "w") as f:
            f.writelines("# counter, sec, nsec, x, y, z, qx, qy, qz, qw\n")
            for i in range(pose.num_poses):
                f.write(str(i) + ", ")
                timestamp_str = TimeStamp(t_float128=pose.timestamps[i]).t_string_sec
                f.write(timestamp_str + ", ")
                timestamp_str = TimeStamp(t_float128=pose.timestamps[i]).t_string_nsec
                f.write(timestamp_str + ", ")
                f.write(str(pose.positions_xyz[i][0]) + ", ")
                f.write(str(pose.positions_xyz[i][1]) + ", ")
                f.write(str(pose.positions_xyz[i][2]) + ", ")
                f.write(str(pose.orientations_quat_wxyz[i][1]) + ", ")
                f.write(str(pose.orientations_quat_wxyz[i][2]) + ", ")
                f.write(str(pose.orientations_quat_wxyz[i][3]) + ", ")
                f.write(str(pose.orientations_quat_wxyz[i][0]))
                f.writelines("\n")
