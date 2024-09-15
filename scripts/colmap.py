from pathlib import Path

from oxford_spires_utils.bash_command import run_command


def run_colmap_mvs(image_path, colmap_output_path, sparse_folder, max_image_size):
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


if __name__ == "__main__":
    image_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw"
    colmap_output_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/processed/output_colmap"
    colmap_output_path = Path(colmap_output_path)
    sparse_folder = colmap_output_path / "sparse" / "0"
    max_image_size = 100
    run_colmap_mvs(image_path, colmap_output_path, sparse_folder, max_image_size)
