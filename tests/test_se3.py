import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from oxford_spires_utils.se3 import (
    quat_xyzw_to_quat_wxyz,
    se3_matrix_to_xyz_quat_wxyz,
    se3_matrix_to_xyz_quat_xyzw,
    xyz_quat_wxyz_to_se3_matrix,
    xyz_quat_xyzw_to_se3_matrix,
)


def test_se3_matrix_to_xyz_quat_xyzw():
    # Test case 1: Identity matrix
    identity = np.eye(4)
    xyz, quat = se3_matrix_to_xyz_quat_xyzw(identity)
    assert np.allclose(xyz, [0, 0, 0])
    assert np.allclose(quat, [0, 0, 0, 1], atol=1e-6)

    # Test case 2: Translation only
    translation = np.eye(4)
    translation[:3, 3] = [1, 2, 3]
    xyz, quat = se3_matrix_to_xyz_quat_xyzw(translation)
    assert np.allclose(xyz, [1, 2, 3])
    assert np.allclose(quat, [0, 0, 0, 1], atol=1e-6)

    # Test case 3: Rotation only (90 degrees around x-axis)
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    xyz, quat = se3_matrix_to_xyz_quat_xyzw(rotation)
    assert np.allclose(xyz, [0, 0, 0])
    assert np.allclose(quat, [0.7071068, 0, 0, 0.7071068], atol=1e-6)

    # Test case 4: Both rotation and translation
    rt = np.eye(4)
    rt[:3, :3] = Rotation.from_euler("xyz", [30, 60, 90], degrees=True).as_matrix()
    rt[:3, 3] = [4, 5, 6]
    xyz, quat = se3_matrix_to_xyz_quat_xyzw(rt)
    assert np.allclose(xyz, [4, 5, 6])
    expected_quat = Rotation.from_euler("xyz", [30, 60, 90], degrees=True).as_quat()
    assert np.allclose(quat, expected_quat, atol=1e-6)


def test_xyz_quat_xyzw_to_se3_matrix():
    # Test case 1: Zero translation, identity rotation
    xyz = [0, 0, 0]
    quat = [0, 0, 0, 1]
    se3 = xyz_quat_xyzw_to_se3_matrix(xyz, quat)
    assert np.allclose(se3, np.eye(4))

    # Test case 2: Translation only
    xyz = [1, 2, 3]
    quat = [0, 0, 0, 1]
    se3 = xyz_quat_xyzw_to_se3_matrix(xyz, quat)
    expected = np.eye(4)
    expected[:3, 3] = xyz
    assert np.allclose(se3, expected)

    # Test case 3: Rotation only (90 degrees around x-axis)
    xyz = [0, 0, 0]
    quat = [0.7071068, 0, 0, 0.7071068]
    se3 = xyz_quat_xyzw_to_se3_matrix(xyz, quat)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    assert np.allclose(se3, expected)

    # Test case 4: Both rotation and translation
    xyz = [4, 5, 6]
    quat = Rotation.from_euler("xyz", [30, 60, 90], degrees=True).as_quat()
    se3 = xyz_quat_xyzw_to_se3_matrix(xyz, quat)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("xyz", [30, 60, 90], degrees=True).as_matrix()
    expected[:3, 3] = xyz
    assert np.allclose(se3, expected)


def test_se3_matrix_to_xyz_quat_wxyz():
    # Test conversion from SE(3) matrix to XYZ + quaternion (WXYZ)
    se3 = np.eye(4)
    se3[:3, 3] = [1, 2, 3]
    se3[:3, :3] = Rotation.from_euler("xyz", [45, 45, 45], degrees=True).as_matrix()
    xyz, quat = se3_matrix_to_xyz_quat_wxyz(se3)
    expected_quat = quat_xyzw_to_quat_wxyz(Rotation.from_euler("xyz", [45, 45, 45], degrees=True).as_quat())
    assert np.allclose(xyz, [1, 2, 3], atol=1e-6)
    assert np.allclose(quat, expected_quat, atol=1e-6)


def test_xyz_quat_wxyz_to_se3_matrix():
    # Test conversion from XYZ + quaternion (WXYZ) to SE(3) matrix
    xyz = [1, 2, 3]
    quat = [1, 0, 0, 0]  # Identity quaternion WXYZ
    se3 = xyz_quat_wxyz_to_se3_matrix(xyz, quat)
    expected = np.eye(4)
    expected[:3, 3] = xyz
    assert np.allclose(se3, expected, atol=1e-6)


def test_roundtrip_conversion():
    # Test that converting from SE3 to xyz-quat and back gives the original matrix
    original_se3 = np.eye(4)
    original_se3[:3, :3] = Rotation.from_euler("xyz", [45, 30, 60], degrees=True).as_matrix()
    original_se3[:3, 3] = [7, 8, 9]

    xyz, quat = se3_matrix_to_xyz_quat_xyzw(original_se3)
    reconstructed_se3 = xyz_quat_xyzw_to_se3_matrix(xyz, quat)

    assert np.allclose(original_se3, reconstructed_se3)


@pytest.mark.parametrize("num_tests", [1000])  # Number of random tests
def test_roundtrip_conversion_rand(num_tests):
    for _ in range(num_tests):
        # Generate a random translation vector (XYZ)
        xyz = np.random.uniform(-10, 10, size=3)

        # Generate a random rotation as a quaternion (XYZW)
        random_quat = Rotation.random().as_quat()  # Generates a random quaternion
        equiv_quat = -random_quat

        # Create the SE(3) matrix from the random translation and quaternion
        original_se3_matrix = xyz_quat_xyzw_to_se3_matrix(xyz, random_quat)
        test_se3_matrix_1 = xyz_quat_xyzw_to_se3_matrix(xyz, equiv_quat)
        np.testing.assert_allclose(original_se3_matrix, test_se3_matrix_1, rtol=1e-5, atol=1e-8)

        # Convert the SE(3) matrix back to (xyz, quat_xyzw)
        extracted_xyz, extracted_quat_xyzw = se3_matrix_to_xyz_quat_xyzw(original_se3_matrix)

        # Check if the extracted XYZ matches the original
        np.testing.assert_allclose(extracted_xyz, xyz, rtol=1e-5, atol=1e-8)
        # Check if the extracted quaternion matches the original quaternion up to sign
        # Since q and -q represent the same rotation, we check both
        assert np.allclose(extracted_quat_xyzw, random_quat, rtol=1e-5, atol=1e-8) or np.allclose(
            extracted_quat_xyzw, -random_quat, rtol=1e-5, atol=1e-8
        ), f"Quaternions do not match: {extracted_quat_xyzw} vs {random_quat} (or its negation)"

        # Reconstruct the SE(3) matrix from the extracted values
        reconstructed_se3_matrix = xyz_quat_xyzw_to_se3_matrix(extracted_xyz, extracted_quat_xyzw)
        # Check if the original and reconstructed SE(3) matrices are the same
        np.testing.assert_allclose(reconstructed_se3_matrix, original_se3_matrix, rtol=1e-5, atol=1e-8)
