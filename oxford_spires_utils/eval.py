import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree as KDTree

logger = logging.getLogger(__name__)


def compute_p2p_distance(query_cloud: np.ndarray, reference_cloud: np.ndarray):
    ref_kd_tree = KDTree(reference_cloud)
    distances, _ = ref_kd_tree.query(query_cloud, workers=-1)
    return distances


def get_recon_metrics(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    precision_threshold=0.05,
    recall_threshold=0.05,
):
    assert isinstance(input_cloud, np.ndarray) and isinstance(gt_cloud, np.ndarray)
    assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3
    logger.info(f"Computing Accuracy and Precision ({precision_threshold}) ...")
    distances = compute_p2p_distance(input_cloud, gt_cloud)
    accuracy = np.mean(distances)
    precision = np.sum(distances < precision_threshold) / len(distances)

    logger.info(f"Computing Completeness and Recall ({recall_threshold}) ...")
    distances = compute_p2p_distance(gt_cloud, input_cloud)
    completeness = np.mean(distances)
    recall = np.sum(distances < recall_threshold) / len(distances)
    f1_score = 2 * (precision * recall) / (precision + recall)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "completeness": completeness,
        "recall": recall,
        "f1_score": f1_score,
    }
    return results


def get_recon_metrics_multi_thresholds(
    input_cloud: np.ndarray,
    gt_cloud: np.ndarray,
    thresholds: list = [0.02, 0.05, 0.1],
):
    assert isinstance(input_cloud, np.ndarray) and isinstance(gt_cloud, np.ndarray)
    assert input_cloud.shape[1] == 3 and gt_cloud.shape[1] == 3
    results = []

    logger.info("Computing Accuracy and Precision ...")
    input_to_gt_dist = compute_p2p_distance(input_cloud, gt_cloud)
    accuracy = np.mean(input_to_gt_dist)

    logger.info("Computing Completeness and Recall ...")
    gt_to_input_dist = compute_p2p_distance(gt_cloud, input_cloud)
    completeness = np.mean(gt_to_input_dist)

    logger.info(f"Accuracy: {accuracy:.4f}, Completeness: {completeness:.4f}")
    results.append({"accuracy": accuracy, "completeness": completeness})
    for threshold in thresholds:
        precision = np.sum(input_to_gt_dist < threshold) / len(input_to_gt_dist)
        recall = np.sum(gt_to_input_dist < threshold) / len(gt_to_input_dist)
        f1_score = 2 * (precision * recall) / (precision + recall)
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )
        logger.info(f"threshold {threshold} m, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1_score:.4f}")
    return results


def save_error_cloud(input_cloud: np.ndarray, reference_cloud: np.ndarray, save_path, cmap="bgyr"):
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

    distances = compute_p2p_distance(input_cloud, reference_cloud)
    distances = np.clip(distances, 0, 1)
    if cmap == "bgyr":
        cmap = get_BGYR_colourmap()
        distances_cmap = cmap(distances / np.max(distances))
    else:
        distances_cmap = plt.get_cmap(cmap)(distances / np.max(distances))

    test_cloud = o3d.geometry.PointCloud()
    test_cloud.points = o3d.utility.Vector3dVector(input_cloud)
    test_cloud.colors = o3d.utility.Vector3dVector(distances_cmap[:, :3])
    o3d.io.write_point_cloud(save_path, test_cloud)
    print(f"Error cloud saved to {save_path}")


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
        print(get_recon_metrics(input_cloud_np, gt_cloud_np))
        save_error_cloud(input_cloud_np, gt_cloud_np, save_error_cloud_path)
