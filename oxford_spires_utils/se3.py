import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_quat_wxyz(quat_xyzw):
    if isinstance(quat_xyzw, list):
        quat_xyzw = np.array(quat_xyzw)
    assert is_quaternion(quat_xyzw), f"{quat_xyzw} is not a valid quaternion"
    return quat_xyzw[[3, 0, 1, 2]]


def se3_matrix_to_xyz_quat_xyzw(se3_matrix):
    assert is_se3_matrix(se3_matrix)[0], f"{se3_matrix} not valid, {is_se3_matrix(se3_matrix)[1]}"
    xyz = se3_matrix[:3, 3]
    quat_xyzw = Rotation.from_matrix(se3_matrix[:3, :3]).as_quat()
    return xyz, quat_xyzw


def se3_matrix_to_xyz_quat_wxyz(se3_matrix):
    assert is_se3_matrix(se3_matrix)[0], f"{se3_matrix} not valid, {is_se3_matrix(se3_matrix)[1]}"
    xyz, quat_xyzw = se3_matrix_to_xyz_quat_xyzw(se3_matrix)
    quat_wxyz = quat_xyzw_to_quat_wxyz(quat_xyzw)
    return xyz, quat_wxyz


def xyz_quat_xyzw_to_se3_matrix(xyz, quat_xyzw):
    se3_matrix = np.eye(4)
    se3_matrix[:3, 3] = xyz
    se3_matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    assert is_se3_matrix(se3_matrix)[0], f"{se3_matrix} not valid, {is_se3_matrix(se3_matrix)[1]}"
    return se3_matrix


def is_se3_matrix(se3_matrix):
    valid_shape = se3_matrix.shape == (4, 4)
    valid_last_row = np.allclose(se3_matrix[3], [0, 0, 0, 1])  # chec k the last row
    R = se3_matrix[:3, :3]
    valid_rot_det = np.isclose(np.linalg.det(R), 1.0, atol=1e-6)  # check the rotation matrix
    valid_orthogonality = np.allclose(R @ R.T, np.eye(3), atol=1e-6)  # check the orthogonality
    is_valid = valid_shape and valid_last_row and valid_rot_det and valid_orthogonality
    debug_info = {
        "valid_shape": valid_shape,
        "valid_last_row": valid_last_row,
        "valid_rot_det": valid_rot_det,
        "valid_orthogonality": valid_orthogonality,
    }
    return is_valid, debug_info


def is_quaternion(quaternion):
    if isinstance(quaternion, list):
        quaternion = np.array(quaternion)
    assert isinstance(quaternion, np.ndarray), f"{quaternion} is not a numpy array or list"
    assert quaternion.shape == (4,), f"{quaternion} is not a 4D quaternion"
    return np.isclose(np.linalg.norm(quaternion), 1.0)
