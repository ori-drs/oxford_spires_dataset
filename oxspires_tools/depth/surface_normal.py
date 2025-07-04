import numpy as np


def compute_normalmap(normals, v, u, h, w, K, D):
    normalmap = np.zeros((h, w, 3), dtype=np.float32)
    if normals.size == 0:
        return normalmap
    assert normals.max() <= 1.0 + 1e-5, normals.max()
    assert normals.min() >= -1.0 - 1e-5, normals.min()

    normalmap[v, u] = normals

    # flip if normal is pointing away from camera
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    x = (i - cx) / fx
    y = (j - cy) / fy
    z = np.ones_like(x)

    pixel_vectors = np.stack((x, y, z), axis=-1)
    cos_theta = np.sum(pixel_vectors * normalmap, axis=-1)
    normalmap[cos_theta > 0] *= -1
    assert normals.max() <= 1.0 + 1e-5, normals.max()
    assert normals.min() >= -1.0 - 1e-5, normals.min()
    normalised_normalmap = ((normalmap + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # hard code empty normal to be 128,128,128
    old_point = np.array([127, 127, 127], dtype=np.uint8)
    new_point = np.array([128, 128, 128], dtype=np.uint8)
    indices = np.where(np.all(normalised_normalmap == old_point, axis=-1))
    normalised_normalmap[indices] = new_point

    return normalised_normalmap
