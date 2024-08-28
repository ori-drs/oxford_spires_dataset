import numpy as np
from oxford_spires_utils.transformations import quaternion_from_matrix, quaternion_matrix


def se3_matrix_to_xyz_quat_xyzw(se3_matrix):
    xyz = se3_matrix[:3, 3]
    quat_xyzw = quaternion_from_matrix(se3_matrix[:3, :3])
    return xyz, quat_xyzw


def xyz_quat_xyzw_to_se3_matrix(xyz, quat_xyzw):
    se3_matrix = np.eye(4)
    se3_matrix[:3, 3] = xyz
    se3_matrix[:3, :3] = quaternion_matrix(quat_xyzw)[:3, :3]
    return se3_matrix
