import evo
import numpy as np
from evo.tools.file_interface import csv_read_matrix

from .base import BasicTrajReader
from .timestamp import TimeStamp


class VilensSlamTrajReader(BasicTrajReader):
    """
    Read VILENS SLAM trajectory file
    """

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path)

    def read_file(self):
        """
        Read VILENS SLAM trajectory file
        @return: PosePath3D from evo
        """
        raw_mat = csv_read_matrix(self.file_path, delim=",", comment_str="#")
        if not raw_mat:
            raise ValueError()
        timestamps = [TimeStamp(sec=row[1], nsec=row[2]).t_float128 for row in raw_mat]
        mat = np.array(raw_mat).astype(float)
        xyz = mat[:, 3:6]
        quat_xyzw = mat[:, 6:10]
        quat_wxyz = np.roll(quat_xyzw, 1, axis=1)  # xyzw -> wxyz
        return evo.core.trajectory.PoseTrajectory3D(xyz, quat_wxyz, timestamps=timestamps)
