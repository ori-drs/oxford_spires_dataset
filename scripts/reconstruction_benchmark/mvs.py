import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from oxford_spires_utils.bash_command import run_command
from oxford_spires_utils.se3 import s_se3_from_sim3
from oxford_spires_utils.trajectory.pose_convention import colmap_to_nerf_world_transform

logger = logging.getLogger(__name__)


def run_colmap_mvs(image_path, colmap_output_path, sparse_folder, max_image_size):
    logger.info(f"Running colmap MVS; img_path {image_path}; output: {colmap_output_path}")
    colmap_image_undistorter_cmd = [
        "colmap image_undistorter",
        f"--image_path {image_path}",
        f"--input_path {sparse_folder}",
        f"--output_path {colmap_output_path/'dense'}",
        "--output_type COLMAP",
        f"--max_image_size {max_image_size}",
    ]
    colmap_image_undistorter_cmd = " ".join(colmap_image_undistorter_cmd)
    run_command(colmap_image_undistorter_cmd, print_command=True)

    colmap_patch_match_stereo_cmd = [
        "colmap patch_match_stereo",
        f"--workspace_path {colmap_output_path/'dense'}",
        "--workspace_format COLMAP",
        "--PatchMatchStereo.geom_consistency true",
    ]
    colmap_patch_match_stereo_cmd = " ".join(colmap_patch_match_stereo_cmd)
    run_command(colmap_patch_match_stereo_cmd, print_command=True)

    colmap_stereo_fusion_cmd = [
        "colmap stereo_fusion",
        f"--workspace_path {colmap_output_path/'dense'}",
        "--workspace_format COLMAP",
        "--input_type geometric",
        f"--output_path {colmap_output_path /'dense'/'fused.ply'}",
    ]
    colmap_stereo_fusion_cmd = " ".join(colmap_stereo_fusion_cmd)
    run_command(colmap_stereo_fusion_cmd, print_command=True)

    colmap_delauany_mesh_filter_cmd = [
        "colmap delaunay_mesher",
        f"--input_path {colmap_output_path /'dense'}",
        f"--output_path {colmap_output_path /'dense'/'meshed-delaunay.ply'}",
    ]
    colmap_delauany_mesh_filter_cmd = " ".join(colmap_delauany_mesh_filter_cmd)
    run_command(colmap_delauany_mesh_filter_cmd, print_command=True)


def run_openmvs(image_path, colmap_output_path, sparse_folder, mvs_dir, openmvs_bin="/usr/local/bin/OpenMVS"):
    logger.info(f"Running OpenMVS; img_path {image_path}; output: {mvs_dir}")
    colmap_output_path = Path(colmap_output_path)
    mvs_dir.mkdir(parents=True, exist_ok=True)

    # Export to openMVS
    export_cmd = [
        f"{openmvs_bin}/InterfaceCOLMAP",
        f"-i {colmap_output_path / 'dense'}",
        "-o scene.mvs",
        f"--image-folder {colmap_output_path / 'dense' / 'images'}",
        f"-w {mvs_dir}",
    ]
    run_command(" ".join(export_cmd), print_command=True)

    # Densify point cloud
    densify_cmd = [
        f"{openmvs_bin}/DensifyPointCloud",
        "scene.mvs",
        f"--dense-config-file {openmvs_bin}/Densify.ini",
        "--resolution-level 1",
        "--number-views 8",
        f"-w {mvs_dir}",
    ]
    run_command(" ".join(densify_cmd), print_command=True)
    output_ply_file = mvs_dir / "scene_dense.ply"
    if not output_ply_file.exists():
        logger.error(f"Failed to generate dense point cloud at {output_ply_file}")
    dense_ply = o3d.io.read_point_cloud(str(output_ply_file))
    dense_ply.transform(colmap_to_nerf_world_transform)
    o3d.io.write_point_cloud(str(output_ply_file.with_name("scene_dense_nerf_world.ply")), dense_ply)
    logger.info("Transformed MVS point cloud to the world frame defined by the nerf convention")
    # Reconstruct the mesh
    # reconstruct_cmd = [f"{openmvs_bin}/ReconstructMesh", "scene_dense.mvs", "-p scene_dense.ply", f"-w {mvs_dir}"]
    # run_command(" ".join(reconstruct_cmd), print_command=True)

    # Refine the mesh
    # refine_cmd = [
    #     f"{openmvs_bin}/RefineMesh",
    #     "scene_dense.mvs",
    #     "-m scene_dense_mesh.ply",
    #     "-o scene_dense_mesh_refine.mvs",
    #     "--scales 1",
    #     "--gradient-step 25.05",
    #     f"-w {mvs_dir}",
    # ]
    # run_command(" ".join(refine_cmd), print_command=True)

    # Texture the mesh
    # texture_cmd = [
    #     f"{openmvs_bin}/TextureMesh",
    #     "scene_dense.mvs",
    #     "-m scene_dense_mesh_refine.ply",
    #     "-o scene_dense_mesh_refine_texture.mvs",
    #     "--decimate 0.5",
    #     f"-w {mvs_dir}",
    # ]
    # run_command(" ".join(texture_cmd), print_command=True)


def rescale_openmvs_cloud(original_cloud_file, sim3_matrix, output_cloud_file):
    cloud = o3d.io.read_point_cloud(str(original_cloud_file))
    scale, se3_matrix = s_se3_from_sim3(sim3_matrix)
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    cloud.transform(scale_matrix)
    cloud.transform(se3_matrix)
    o3d.io.write_point_cloud(str(output_cloud_file), cloud)
    logger.info(f"Rescaled OpenMVS point cloud to to metric and save as {output_cloud_file}")


if __name__ == "__main__":
    image_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw"
    colmap_output_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/processed/output_colmap"
    colmap_output_path = Path(colmap_output_path)
    sparse_folder = colmap_output_path / "sparse" / "0"
    max_image_size = 100
    # run_colmap_mvs(image_path, colmap_output_path, sparse_folder, max_image_size)
    mvs_dir = colmap_output_path / "mvs"
    openmvs_bin = "/usr/local/bin/OpenMVS"
    run_openmvs(image_path, colmap_output_path, sparse_folder, mvs_dir, max_image_size, openmvs_bin)
