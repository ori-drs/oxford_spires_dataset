import numpy as np

from oxspires_tools.se3 import is_se3_matrix


def test_is_se3_matrix_rejects_3x3_without_index_error():
    valid, info = is_se3_matrix(np.eye(3))

    assert valid is False
    assert info == {
        "valid_shape": False,
        "valid_last_row": False,
        "valid_rot_det": False,
        "valid_orthogonality": False,
    }


def test_is_se3_matrix_rejects_non_square_4x3_without_linalg_error():
    valid, info = is_se3_matrix(np.zeros((4, 3)))

    assert valid is False
    assert info["valid_shape"] is False


def test_is_se3_matrix_rejects_non_array_input_cleanly():
    valid, info = is_se3_matrix([[1, 0, 0, 0]] * 4)

    assert valid is False
    assert info["valid_shape"] is False
