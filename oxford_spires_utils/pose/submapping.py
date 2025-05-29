import os
import shutil
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from sklearn.cluster import SpectralClustering

from oxford_spires_utils.depth.lidar_fisheye_projection import get_depth_from_cloud
from oxford_spires_utils.depth.mesh_depth import get_depth_from_mesh, get_vertices_visibility_from_mesh
from oxford_spires_utils.depth.utils import save_projection_outputs
from oxford_spires_utils.pose.utils import PosePlotter


class SubmapHandler:
    def __init__(self, poses: PoseTrajectory3D):
        """
        Partition the map into submaps using normalised cut / spectral clustering
        @param poses: PoseTrajectory3D
        """
        self.poses = poses

    def run_spectral_clustering(
        self, similarity_matrix, n_clusters=7, plot_similarity_matrix=False, min_similarity_viz=0.5
    ):
        """
        Partition the map into submaps using normalised cut / spectral clustering
        @param similarity_matrix: N_img x N_img similarity matrix
        @param n_cluster: number of clusters
        """
        if plot_similarity_matrix:
            pose_plotter = PosePlotter(
                self.poses, similarity_matrix, pose_viz_format="axis", axis_viz_size=0.3, min_sim=min_similarity_viz
            )
            pose_plotter.plot_poses()
        clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
        clustering.fit(similarity_matrix)
        labels = clustering.labels_
        return labels

    def viz_submap_cluster(
        self,
        labels,
        save_path="submap_cluster.png",
    ):
        unique_labels = np.unique(labels)
        colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        plt.figure()
        for label in unique_labels:
            color = colors[label % len(colors)]
            cluster_poses = self.poses.positions_xyz[labels == label]
            plt.scatter(cluster_poses[:, 0], cluster_poses[:, 1], color=color, label=f"submap_{label}")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Submap Clustering")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


def compute_image_mesh_visibility(map_mesh, fnt_dataset, depth_dir=None):
    visibility_matrices = {}

    for camera in fnt_dataset.sensor.cameras:
        camera_name = camera.label
        # K, D, h, w, fov_deg, camera_model = fnt_dataset.sensor.get_params_for_depth(camera_name)
        # visibility_matrices[camera.label] = np.zeros(
        #     (fnt_dataset.vilens_slam_handler.num_poses, len(map_cloud.points)), dtype=bool
        # )
        depth_subdir = Path(depth_dir) / camera.label if depth_dir is not None else None
        depth_subdir.mkdir(parents=True, exist_ok=True)
        map_mesh_vertices = map_mesh.primitives[0].positions
        import open3d as o3d

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(map_mesh_vertices)
        voxel_size = 0.5
        o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=voxel_size)
        # map_mesh_vertices = np.asarray(o3d_cloud.points)
        visibility_list = []
        for i in range(fnt_dataset.vilens_slam_handler.num_poses):
            print(i)
            if fnt_dataset.image_paths[camera.label][i] is not None:
                T_world_cam = fnt_dataset.get_pose(i, camera_name, source="slam")
                depth = get_depth_from_mesh(map_mesh, T_world_cam, save_path=depth_subdir / f"{i}.png")
                visibility_mask, depth_image = get_vertices_visibility_from_mesh(map_mesh_vertices, depth, T_world_cam)
                current_cloud = deepcopy(o3d_cloud)
                current_cloud.transform(np.linalg.inv(T_world_cam))
                # depth_vertices, normal, cloud_mask = get_depth_from_cloud(
                #     current_cloud, K, D, w, h, fov_deg, camera_model, compute_cloud_mask=True
                # )
                depth_vertices_normalised = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
                import cv2

                depth_mask = depth_vertices_normalised > 0.0
                depth_vertices = (depth_vertices_normalised * 255).astype(np.uint8)
                depth_vertices = cv2.applyColorMap(depth_vertices, cv2.COLORMAP_PLASMA)
                depth_vertices[~depth_mask] = 255
                cv2.imwrite(str(depth_subdir / f"vertices_{i}.jpg"), depth_vertices)
                visibility_list.append(visibility_mask)
                # save_projection_outputs(depth_vertices, save_depth_path=depth_subdir / f"vertices_{i}.jpg")
        visibility_matrices[camera.label] = np.array(visibility_list)
    return visibility_matrices


def compute_image_point_visibility(map_cloud, fnt_dataset, overlay_dir=None):
    visibility_matrices = {}
    for camera in fnt_dataset.sensor.cameras:
        camera_name = camera.label
        K, D, h, w, fov_deg, camera_model = fnt_dataset.sensor.get_params_for_depth(camera_name)
        # visibility_matrices[camera.label] = np.zeros(
        #     (fnt_dataset.vilens_slam_handler.num_poses, len(map_cloud.points)), dtype=bool
        # )
        visibility_list = []
        overlay_subdir = Path(overlay_dir) / camera.label if overlay_dir is not None else None
        overlay_subdir.mkdir(parents=True, exist_ok=True)
        for i in range(fnt_dataset.vilens_slam_handler.num_poses):
            if fnt_dataset.image_paths[camera.label][i] is not None:
                image_path = fnt_dataset.image_paths[camera.label][i]
                T_world_cam = fnt_dataset.get_pose(i, camera_name, source="slam")
                T_cam_world = np.linalg.inv(T_world_cam)
                map_cloud_local = deepcopy(map_cloud)
                map_cloud_local.transform(T_cam_world)

                depth, normal, cloud_mask = get_depth_from_cloud(
                    map_cloud_local, K, D, w, h, fov_deg, camera_model, compute_cloud_mask=True
                )
                save_projection_outputs(depth, normal, image_path, save_overlay_path=overlay_subdir / f"{i}.jpg")

                visibility_list.append(cloud_mask)
        visibility_matrices[camera.label] = np.array(visibility_list)
        # visibility_matrices[camera.label][i] = point_mask

    return visibility_matrices


def compute_similarity_matrix(visibility_matrices, save_path="similarity_matrix.png"):
    """
    Compute similarity matrix between camera sequences
    @param visibility_matrices: N_img x N_pts boolean matrices
    @return: N_img x N_img similarity matrix
    """
    num_images = visibility_matrices.shape[0]
    # num_points = visibility_matrices.shape[1]
    similarity_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i, num_images):
            similarity_matrix[i, j] = np.sum(visibility_matrices[i] & visibility_matrices[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    # plot similarity matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(18, 18))
    ax = sns.heatmap(similarity_matrix, annot=False, cmap="viridis", cbar=True, square=True)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.title("Camera Similarity Matrix")
    plt.xlabel("Camera Index")
    plt.ylabel("Camera Index")
    # save
    plt.savefig(save_path)

    return similarity_matrix


def create_submap_image_folders(
    image_paths,
    labels,
    save_dir,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in np.unique(labels):
        submap_dir = save_dir / f"submap_{label}"
        # remove existing submap
        if submap_dir.exists():
            for it in submap_dir.glob("*"):
                if it.is_symlink():
                    it.unlink()
                elif it.is_dir():
                    shutil.rmtree(it)
                else:
                    os.remove(it)
        submap_dir.mkdir(parents=True, exist_ok=True)
        for i, image_path in enumerate(image_paths):
            if labels[i] == label:
                new_image_path = submap_dir / (image_path.parent.name + "_" + image_path.name)
                new_image_path.parent.mkdir(parents=True, exist_ok=True)
                new_image_path.symlink_to(image_path)
