import numpy as np
from numpy.testing import assert_array_almost_equal

from oxford_spires_utils.utils import get_nerf_pose


def colmap_to_nerf(c2w):
    c2w = c2w.copy()  # Create a copy to avoid modifying the original
    c2w[0:3, 2] *= -1  # flip the y and z axis
    c2w[0:3, 1] *= -1
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[2, :] *= -1  # flip whole world upside down
    return c2w


def test_identity_transform():
    identity = np.eye(4)
    result = get_nerf_pose(identity)
    expected = colmap_to_nerf(identity)
    assert_array_almost_equal(result, expected)


def test_translation_only():
    translation = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    result = get_nerf_pose(translation)
    expected = colmap_to_nerf(translation)
    assert_array_almost_equal(result, expected)


def test_rotation_only():
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    result = get_nerf_pose(rotation)
    expected = colmap_to_nerf(rotation)
    assert_array_almost_equal(result, expected)


def test_rotation_and_translation():
    transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    result = get_nerf_pose(transform)
    expected = colmap_to_nerf(transform)
    assert_array_almost_equal(result, expected)


def test_arbitrary_transform():
    transform = np.array([[0.866, -0.5, 0, 1], [0.5, 0.866, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    result = get_nerf_pose(transform)
    expected = colmap_to_nerf(transform)
    assert_array_almost_equal(result, expected)
