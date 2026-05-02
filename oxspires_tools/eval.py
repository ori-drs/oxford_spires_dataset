import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree as KDTree

logger = logging.getLogger(__name__)


def compute_p2p_distance(query_cloud: np.ndarray, reference_cloud: np.ndarray, max_distance=np.inf):
    ref_kd_tree = KDTree(reference_cloud)
    distances, _ = ref_kd_tree.query(query_cloud, workers=-1, distance_upper_bound=max_distance)
    return distances


def get_threshold_ratio(error_array, threshold):
    assert isinstance(error_array, np.ndarray)
    assert error_array.ndim == 1
    assert error_array.shape[0] > 0
    assert isinstance(threshold, (int, float))
    assert threshold > 0

    return np.sum(error_array < threshold) / len(error_array)


def compute_f1_score(precision: float, recall: float):
    assert precision >= 0 and precision <= 1
    assert recall >= 0 and recall <= 1
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def get_distances(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    compute_precision=True,
    compute_recall=True,
    max_distance=np.inf,
):
    assert isinstance(input_cloud, np.ndarray) and isinstance(gt_cloud, np.ndarray)
    assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3
    if input_cloud.shape[0] == 0:
        logger.warning("Input cloud is empty")
        return None
    if gt_cloud.shape[0] == 0:
        logger.warning("GT cloud is empty")
        return None
    distance_precision = None
    distance_recall = None
    if compute_precision:
        distance_precision = compute_p2p_distance(input_cloud, gt_cloud, max_distance=max_distance)
    if compute_recall:
        distance_recall = compute_p2p_distance(gt_cloud, input_cloud, max_distance=max_distance)
    return distance_precision, distance_recall


def get_recon_metrics_from_dist(
    distances_acc: np.ndarray,
    distances_cmpl: np.ndarray,
    precision_threshold=0.05,
    recall_threshold=0.05,
):
    accuracy, precision, completeness, recall, f1_score = None, None, None, None, None
    if distances_acc is not None:
        assert isinstance(distances_acc, np.ndarray)
        assert distances_acc.ndim == 1
        assert distances_acc.shape[0] > 0
        # assert distances_acc >= 0
        assert np.all(distances_acc >= 0)
        accuracy = np.mean(distances_acc)
        precision = get_threshold_ratio(distances_acc, precision_threshold)
    if distances_cmpl is not None:
        assert isinstance(distances_cmpl, np.ndarray)
        assert distances_cmpl.ndim == 1
        assert distances_cmpl.shape[0] > 0
        assert np.all(distances_cmpl >= 0)
        completeness = np.mean(distances_cmpl)
        recall = get_threshold_ratio(distances_cmpl, recall_threshold)
    if distances_acc is not None and distances_cmpl is not None:
        f1_score = compute_f1_score(precision, recall)
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "completeness": completeness,
        "recall": recall,
        "f1_score": f1_score,
    }
    return results


def get_recon_metrics(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    precision_threshold=0.05,
    recall_threshold=0.05,
    compute_precision=True,
    compute_recall=True,
    save_error_cloud_dir=None,
    csv_path=None,
    max_distance=np.inf,
):
    assert isinstance(input_cloud, np.ndarray) and isinstance(gt_cloud, np.ndarray)
    assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3
    if input_cloud.shape[0] == 0:
        logger.warning("Input cloud is empty")
        return None

    distance_precision, distance_recall = get_distances(
        input_cloud,
        gt_cloud,
        compute_precision=compute_precision,
        compute_recall=compute_recall,
        max_distance=max_distance,
    )
    results = get_recon_metrics_from_dist(
        distances_acc=distance_precision,
        distances_cmpl=distance_recall,
        precision_threshold=precision_threshold,
        recall_threshold=recall_threshold,
    )

    if save_error_cloud_dir:
        if compute_precision:
            save_error_cloud(
                input_cloud,
                Path(save_error_cloud_dir) / "input_error_cloud.ply",
                distances=distance_precision,
                max_distance=max_distance,
            )
        # if compute_recall:
        #     save_error_cloud(gt_cloud, Path(save_error_cloud_dir) / f"gt_error_cloud.ply", distance_recall, max_distance)
    if csv_path:
        if Path(csv_path).exists():
            Path(csv_path).unlink()
        dict_to_csv({**results}, csv_path)

    return results


def dict_to_csv(dict_data, filename):
    with open(filename, "a", newline="") as output_file:
        dict_writer = csv.writer(output_file)
        dict_writer.writerow(dict_data.keys())
        dict_writer.writerow(dict_data.values())


def get_BGYR_colourmap():
    colours = [
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green (as specified)
        (255, 255, 0),  # Yellow (as specified)
        (255, 0, 0),  # Red
    ]
    colours = [(r / 255, g / 255, b / 255) for r, g, b in colours]

    # Create the custom colormap
    n_bins = 100  # Number of color segments
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colours, N=n_bins)
    return cmap


def save_error_cloud(
    input_cloud_np: np.ndarray,
    save_path,
    reference_cloud_np: np.ndarray = None,
    distances=None,
    cmap="bgyr",
    max_distance=np.inf,
):
    assert isinstance(input_cloud_np, np.ndarray)
    assert (isinstance(distances, np.ndarray) and input_cloud_np.shape[0] == distances.shape[0]) or isinstance(
        reference_cloud_np, np.ndarray
    )
    if input_cloud_np.shape[0] == 0:
        logger.warning("Input cloud is empty")
        return
    if distances is None:
        distances = compute_p2p_distance(input_cloud_np, reference_cloud_np)
    input_cloud_np = input_cloud_np[distances <= max_distance]
    distances = distances[distances <= max_distance]
    distances = np.clip(distances, 0, 1)
    if cmap == "bgyr":
        cmap = get_BGYR_colourmap()
        distances_cmap = cmap(distances / np.max(distances))
    else:
        distances_cmap = plt.get_cmap(cmap)(distances / np.max(distances))

    test_cloud = o3d.geometry.PointCloud()
    test_cloud.points = o3d.utility.Vector3dVector(input_cloud_np)
    test_cloud.colors = o3d.utility.Vector3dVector(distances_cmap[:, :3])
    o3d.io.write_point_cloud(str(save_path), test_cloud)
    logger.debug(f"diff cloud saved to {save_path}")


if __name__ == "__main__":
    gt_cloud_path = "/home/yifu/data/gt/maths-ts-161019-final.ply"
    gt_cloud = o3d.io.read_point_cloud(gt_cloud_path)

    input_cloud_paths = [
        "/home/yifu/workspace/silvr/exported_clouds/submap_2024-07-14_132335.ply",
        "/home/yifu/workspace/silvr/exported_clouds/submap_2024-07-14_134014.ply",
        "/home/yifu/workspace/silvr/exported_clouds/submap_2024-07-14_135719.ply",
    ]

    for input_cloud_path in input_cloud_paths:
        input_cloud = o3d.io.read_point_cloud(input_cloud_path)
        save_error_cloud_path = str(Path(input_cloud_path).parent / (Path(input_cloud_path).stem + "_error.ply"))
        input_cloud_np = np.array(input_cloud.points)
        gt_cloud_np = np.array(gt_cloud.points)
        logger.info(get_recon_metrics(input_cloud_np, gt_cloud_np))
        save_error_cloud(input_cloud_np, gt_cloud_np, save_error_cloud_path)
