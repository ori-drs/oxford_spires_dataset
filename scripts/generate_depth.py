import argparse
import logging
import shutil
from pathlib import Path

import open3d as o3d
import yaml
from tqdm.auto import tqdm

from oxspires_tools.depth.main import get_depth_from_cloud
from oxspires_tools.depth.utils import save_projection_outputs
from oxspires_tools.sensor import Sensor
from oxspires_tools.utils import get_image_pcd_sync_pair, setup_logging

logger = logging.getLogger(__name__)


def setup_output_dirs(
    proj_dir: Path,
    depth_dir: str,
    normal_dir: str,
    camera_subdirs: list,
    image_folder_path: str,
    save_overlay: bool,
    save_normal_map: bool,
):
    """Create all output directories (top-level and per-camera), removing any existing ones."""
    overlay_dir = proj_dir / (depth_dir + "_overlay")
    output_depth_dir = proj_dir / depth_dir
    shutil.rmtree(output_depth_dir, ignore_errors=True)
    if save_overlay:
        shutil.rmtree(overlay_dir, ignore_errors=True)

    output_normal_dir = None
    if save_normal_map:
        output_normal_dir = proj_dir / normal_dir
        shutil.rmtree(output_normal_dir, ignore_errors=True)

    target_subdirs = {}
    for subdir in camera_subdirs:
        (output_depth_dir / subdir).mkdir(parents=True, exist_ok=True)
        if save_normal_map:
            (output_normal_dir / subdir).mkdir(parents=True, exist_ok=True)
        if save_overlay:
            (overlay_dir / subdir).mkdir(parents=True, exist_ok=True)
        target_subdirs[subdir] = (
            Path(image_folder_path) / subdir,
            output_depth_dir / subdir,
            output_normal_dir / subdir if save_normal_map else None,
            overlay_dir / subdir if save_overlay else None,
        )

    return output_depth_dir, output_normal_dir, overlay_dir, target_subdirs


def project_lidar_to_fisheye(
    sensor,
    project_dir: str,
    depth_dir: str,
    normal_dir: str,
    image_folder_path: str,
    depth_pose_format: str,
    slam_individual_clouds_new_path: str,
    is_euclidean: bool,
    max_time_diff_camera_and_pose: float,
    depth_pose_path: str = None,
    image_ext: str = ".jpg",
    depth_factor: float = 256.0,
    save_overlay: bool = True,
    save_normal_map: bool = True,
    camera_model: str = "OPENCV_FISHEYE",
):
    logger.info("Depth is euclidean: L2 distance between points and camera" if is_euclidean else "Depth is not euclidean: z_value")  # fmt: skip
    image_ext = f".{image_ext}" if not image_ext.startswith(".") else image_ext

    _, _, _, target_subdirs = setup_output_dirs(
        Path(project_dir),
        depth_dir,
        normal_dir,
        list(sensor.camera_topics_labelled.values()),
        image_folder_path,
        save_overlay,
        save_normal_map,
    )

    for cam_name, subdir in sensor.camera_topics_labelled.items():
        logger.info(f"Processing {cam_name} in {subdir} ...")
        target_image_subdir, target_depth_subdir, target_normal_subdir, target_overlay_subdir = target_subdirs[subdir]
        K, D, h, w, fov_deg, _ = sensor.get_params_for_depth(cam_name, depth_pose_format, depth_pose_path)
        logger.info(f"Fov: {fov_deg}")
        T_cam_base = sensor.tf.get_transform("base", cam_name)
        image_pcd_pairs = get_image_pcd_sync_pair(target_image_subdir, slam_individual_clouds_new_path, image_ext, max_time_diff_camera_and_pose)  # fmt: skip

        for image_path, pcd_path, _ in tqdm(image_pcd_pairs):
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if pcd is None or len(pcd.points) == 0:
                logger.warning(f"Skipping {pcd_path}: {'failed to read' if pcd is None else 'empty'}")
                continue
            pcd.transform(T_cam_base)
            depth, normal = get_depth_from_cloud(pcd, K, D, w, h, fov_deg, camera_model, depth_factor, is_euclidean)
            save_projection_outputs(
                depth,
                normal,
                image_path,
                save_depth_path=target_depth_subdir / (image_path.stem + ".png"),
                save_normal_path=target_normal_subdir / (image_path.stem + ".png") if target_normal_subdir else None,
                save_overlay_path=target_overlay_subdir / (image_path.stem + ".jpg") if target_overlay_subdir else None,
            )


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate depth maps from LiDAR point clouds")
    parser.add_argument(
        "--sequence_dir",
        type=str,
        default="data/sequences/2024-03-18-christ-church-01",
        help="Path to the sequence directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = get_args()

    config_yaml_path = Path(__file__).parent.parent / "configs" / "sensor.yaml"
    with open(config_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    sensor = Sensor(**yaml_data["sensor"])

    seq_dir = Path(args.sequence_dir)

    project_lidar_to_fisheye(
        sensor=sensor,
        project_dir=str(seq_dir / "processed" / "oxspires_tools_outputs"),
        depth_dir="depths_euc_accum_0",
        normal_dir="normals_euc_accum_0",
        image_folder_path=str(seq_dir / "processed" / "colmap"),
        depth_pose_format="vilens_slam",
        slam_individual_clouds_new_path=str(seq_dir / "processed" / "vilens-slam" / "undist-clouds"),
        is_euclidean=True,
        max_time_diff_camera_and_pose=0.025,
        image_ext=".jpg",
        depth_factor=256.0,
        save_overlay=True,
        save_normal_map=True,
        camera_model="OPENCV_FISHEYE",
    )
