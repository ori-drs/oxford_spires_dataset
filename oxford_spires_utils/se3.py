import numpy as np
from scipy.spatial.transform import Rotation


def se3_matrix_to_xyz_quat_xyzw(se3_matrix):
    xyz = se3_matrix[:3, 3]
    quat_xyzw = Rotation.from_matrix(se3_matrix[:3, :3]).as_quat()
    return xyz, quat_xyzw


def xyz_quat_xyzw_to_se3_matrix(xyz, quat_xyzw):
    se3_matrix = np.eye(4)
    se3_matrix[:3, 3] = xyz
    se3_matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return se3_matrix
