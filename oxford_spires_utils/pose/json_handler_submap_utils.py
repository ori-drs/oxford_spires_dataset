import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def get_se3(frame):
    return np.array(frame["transform_matrix"])


def get_xyz(frame):
    se3 = get_se3(frame)
    return se3[:3, 3]


def viz_submap_cluster(labels, points, size=1, save_path=None, h=8, w=12):
    unique_labels = np.unique(labels)
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    # set size
    plt.figure(figsize=(w, h))
    for label in unique_labels:
        color = colors[label % len(colors)]
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=size, color=color, label=f"submap_{label}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Submap Clustering")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def viz_matrix(matrix, max_value=None, save_path=None):
    colourmap = plt.cm.get_cmap("viridis")
    if max_value is not None:
        matrix = np.clip(matrix, 0, max_value)

    plt.imshow(matrix, cmap=colourmap)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def viz_submap_affinity_matrix(points, adjacency_matrix, save_path=None, min_adjacency=0):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(points)
    # plot lines with similarity matrix
    lines = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] >= min_adjacency:
                lines.append([i, j])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([o3d_cloud, line_set])


def get_extra_poses(clusters, current_label, full_xyz_list, overlap_dist=0):
    new_clusters = clusters.copy()
    current_cluster_xyzs = full_xyz_list[clusters == current_label]
    for i, xyz in enumerate(full_xyz_list):
        smallest_dist_to_cluster = np.min(np.linalg.norm(current_cluster_xyzs - xyz, axis=1))
        if smallest_dist_to_cluster < overlap_dist:
            new_clusters[i] = current_label
    return new_clusters


def save_submap_cluster(traj, clusters, save_dir, xyz_list=None, overlap_dist=0, viz=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = np.unique(clusters)

    for label in unique_labels:
        output_traj = traj.copy()
        submap_poses = [frame for frame, m in zip(traj["frames"], clusters == label) if m]
        if viz:
            new_clusters = get_extra_poses(clusters, label, xyz_list, overlap_dist=overlap_dist)
            viz_submap_cluster(new_clusters, xyz_list, save_path=save_dir / f"submap_{label}.png")

        output_traj["frames"] = submap_poses
        save_path = save_dir / f"submap_{label}.json"
        with open(save_path, "w") as f:
            json.dump(output_traj, f, indent=4)


def merge_json(input_folder, output_file_name="transforms_merged.json"):
    # get all json files in input_folder
    input_jsons = list(input_folder.glob("*.json"))

    merged_traj = []
    for file_path in input_jsons:
        with open(file_path, "r") as f:
            traj = json.load(f)
            traj = traj["frames"]
            merged_traj += traj

    final_traj = {"frames": merged_traj}
    with open(output_file_name, "w") as f:
        json.dump(final_traj, f, indent=4)


def combine_cloud(
    pcd_files,
    save_path,
    down_sample=False,
):
    # pcd_files = Path(cloud_dir).glob("*.pcd")
    combine_cloud = o3d.geometry.PointCloud()
    for pcd_file in pcd_files:
        cloud = o3d.io.read_point_cloud(str(pcd_file))
        if down_sample:
            cloud = cloud.voxel_down_sample(voxel_size=0.01)
        combine_cloud += cloud
    o3d.io.write_point_cloud(str(save_path), combine_cloud)


def combine_submap_clouds(
    submap_cloud_dir,
    down_sample=False,
):
    submap_cloud_dir = Path(submap_cloud_dir)
    camera_dirs = submap_cloud_dir.glob("*")
    pcd_files = []
    for camera_dir in camera_dirs:
        pcd_files += list(Path(camera_dir).glob("*.pcd"))
    save_path = submap_cloud_dir / f"{submap_cloud_dir}_lidar.pcd"
    combine_cloud(
        pcd_files=pcd_files,
        save_path=save_path,
        down_sample=down_sample,
    )
