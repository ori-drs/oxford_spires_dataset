import cv2
import numpy as np


def _camera_rays_from_pixels(u, v, K, D, camera_model):
    """Back-project image pixels to camera-frame rays using the calibrated camera model."""
    pixels = np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2)
    if camera_model == "OPENCV_FISHEYE":
        normalised = cv2.fisheye.undistortPoints(pixels, K, D, P=np.eye(3))
    elif camera_model == "OPENCV":
        normalised = cv2.undistortPoints(pixels, K, D, P=np.eye(3))
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")
    normalised = normalised.reshape(-1, 2)
    return np.column_stack((normalised, np.ones(len(normalised))))


def compute_normalmap(normals, v, u, h, w, K, D, camera_model="OPENCV_FISHEYE"):
    normalmap = np.zeros((h, w, 3), dtype=np.float32)
    if normals.size == 0:
        return normalmap
    assert normals.max() <= 1.0 + 1e-5, normals.max()
    assert normals.min() >= -1.0 - 1e-5, normals.min()

    # Orient each normal toward the camera using the actual calibrated image ray.
    # For fisheye cameras the pinhole approximation (u-cx)/fx, (v-cy)/fy, 1
    # does not represent the projected ray away from the image centre.
    oriented_normals = normals.copy()
    pixel_vectors = _camera_rays_from_pixels(u, v, K, D, camera_model)
    cos_theta = np.sum(pixel_vectors * oriented_normals, axis=1)
    oriented_normals[cos_theta > 0] *= -1

    # Points are already depth-sorted far-to-near before this call, so duplicate
    # pixels retain the nearest point's normal through NumPy's final assignment.
    normalmap[v, u] = oriented_normals

    assert oriented_normals.max() <= 1.0 + 1e-5, oriented_normals.max()
    assert oriented_normals.min() >= -1.0 - 1e-5, oriented_normals.min()
    normalised_normalmap = ((normalmap + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # Hard-code empty normal to be 128,128,128.
    old_point = np.array([127, 127, 127], dtype=np.uint8)
    new_point = np.array([128, 128, 128], dtype=np.uint8)
    indices = np.where(np.all(normalised_normalmap == old_point, axis=-1))
    normalised_normalmap[indices] = new_point

    return normalised_normalmap
