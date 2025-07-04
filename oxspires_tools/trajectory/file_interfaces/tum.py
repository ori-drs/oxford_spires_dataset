"""
TUM format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
'timestamp tx ty tz qx qy qz qw'
"""

import os

import evo
import numpy as np

from .base import BasicTrajReader, BasicTrajWriter
from .timestamp import TimeStamp


class TUMTrajReader(BasicTrajReader):
    """
    Read TUM trajectory file
    """

    def __init__(
        self, file_path, tum_reader_type="custom", tum_custom_reader_prefix="", tum_custom_reader_suffix="", **kwargs
    ):
        super().__init__(file_path)
        assert tum_reader_type in ["evo", "custom"], "reader_type should be either 'evo' or 'custom'"
        self.reader_type = tum_reader_type
        self.custom_reader_prefix = tum_custom_reader_prefix
        self.custom_reader_suffix = tum_custom_reader_suffix

    def read_file(self):
        """
        Read TUM trajectory file
        @return: PosePath3D from evo
        """
        if self.reader_type == "evo":
            print("Try to load TUM pose using evo; Assuming file name is timestamp")
            tum_pose = evo.tools.file_interface.read_tum_trajectory_file(self.file_path)
            assert tum_pose.check()[0] is True, tum_pose.check()[1]
        elif self.reader_type == "custom":
            print("Try to load TUM pose using custom reader; Assuming file name is not timestamp")
            tum_pose = self.read_tum_pose_custom(
                self.file_path, prefix=self.custom_reader_prefix, suffix=self.custom_reader_suffix
            )
            assert tum_pose.check()[0] is True, tum_pose.check()[1]
        else:
            raise ValueError("reader_type should be either 'evo' or 'custom'")
        return tum_pose

    def read_tum_pose_custom(self, file_path, prefix="", suffix=""):
        """
        this is used for custum TUM style trajectory file but timestamp has custom prefix and suffix
        format: prefix<secs.nsecs>suffix x y z qx qy qz qw (NOTE the quaternion order)
        """

        if prefix != "" or suffix != "":
            print(f'Assuming file name is timestamp in the format: "{prefix}<secs.nsecs>{suffix}"')
        else:
            print(
                "If no prefix/suffix provide, assuming fname is timestamp"  # ,\
                # and you could use evo.tools.file_interface.read_tum_trajectory_file() instead"
            )
        pose_file = open(file_path, "r")

        xyz = []
        quat = []
        time_stamp = []
        for pose_line in pose_file:
            if pose_line[0] == "#":
                continue
            splited = pose_line.rstrip().split(" ")
            assert len(splited) == 8, "Each line should have 8 elements, but got {}".format(len(splited))

            fname = splited[0]
            if fname[: len(prefix)] != prefix or fname[len(fname) - len(suffix) :] != suffix:
                print("skipping lines with wrong prefix or suffix")
                continue
            t_float128 = TimeStamp(t_string=fname[len(prefix) : len(fname) - len(suffix)]).t_float128
            assert TimeStamp.get_string_from_t_float128(t_float128) == fname[len(prefix) : len(fname) - len(suffix)], (
                f"loss of precision in timestamp: before {fname[len(prefix) : len(fname) - len(suffix)]}; after {t_float}"
            )
            time_stamp.append(t_float128)
            translation = splited[1:4]
            quaternion_xyzw = splited[4:8]
            quaternion_wxyz = [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
            translation = [float(_) for _ in translation]
            quaternion_wxyz = [float(_) for _ in quaternion_wxyz]
            xyz.append(translation)
            quat.append(quaternion_wxyz)
        xyz = np.array(xyz)
        quat = np.array(quat)
        timestamps = np.array(time_stamp)

        return evo.core.trajectory.PoseTrajectory3D(xyz, quat, timestamps=timestamps)


class TUMTrajWriter(BasicTrajWriter):
    """
    Write trajectory file in TUM format
    """

    def __init__(self, file_path, **kwargs):
        super().__init__(file_path)

    def write_file(self, pose):
        self.write_tum_pose_custom(self.file_path, pose, prefix="", suffix="")

    def write_tum_pose_custom(self, file_path, pose, prefix="", suffix=""):
        """
        this is used for custum TUM style trajectory file but timestamp has custom prefix and suffix
        @param file_path: path to save the file
        @param pose: PoseTrajectory3D from evo
        @param prefix: prefix of filename to be added before timestamp
        @param suffix: suffix of filename to be added after timestamp
        """
        if prefix != "" or suffix != "":
            print(f'Assuming file name is timestamp in the format: "{prefix}<secs.nsecs>{suffix}"')
        if not isinstance(pose, evo.core.trajectory.PoseTrajectory3D):
            raise ValueError("pose should be PoseTrajectory3D from evo")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        assert pose.check()[0] is True, pose.check()[1]

        print("Writing pose message in TUM format to: ", file_path)
        with open(file_path, "w") as f:
            f.writelines("# timestamp tx ty tz qx qy qz qw\n")
            for i in range(pose.num_poses):
                timestamp_str = TimeStamp(t_float128=pose.timestamps[i]).t_string
                f.write(prefix + timestamp_str + suffix + " ")
                f.write(str(pose.positions_xyz[i][0]) + " ")
                f.write(str(pose.positions_xyz[i][1]) + " ")
                f.write(str(pose.positions_xyz[i][2]) + " ")
                f.write(str(pose.orientations_quat_wxyz[i][1]) + " ")
                f.write(str(pose.orientations_quat_wxyz[i][2]) + " ")
                f.write(str(pose.orientations_quat_wxyz[i][3]) + " ")
                f.write(str(pose.orientations_quat_wxyz[i][0]))
                f.writelines("\n")
