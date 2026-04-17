import logging
import shutil
from pathlib import Path

import open3d as o3d
import yaml
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from oxspires_tools.depth.main import get_depth_from_cloud
from oxspires_tools.depth.utils import save_projection_outputs
from oxspires_tools.sensor import Sensor
from oxspires_tools.utils import get_image_pcd_sync_pair, unzip_files

logger = logging.getLogger(__name__)


def setup_output_dirs(
    proj_dir: Path,
    depth_dir: str,
    normal_dir: str,
    camera_subdirs: list,
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

    for subdir in camera_subdirs:
        (output_depth_dir / subdir).mkdir(parents=True, exist_ok=True)
        if save_normal_map:
            (output_normal_dir / subdir).mkdir(parents=True, exist_ok=True)
        if save_overlay:
            (overlay_dir / subdir).mkdir(parents=True, exist_ok=True)

    return output_depth_dir, output_normal_dir, overlay_dir


def preprocess_camera(
    sensor,
    cam_name: str,
    subdir: str,
    image_folder_path: str,
    slam_individual_clouds_new_path: str,
    output_depth_dir: Path,
    output_normal_dir: Path,
    overlay_dir: Path,
    depth_pose_format: str,
    depth_pose_path: str,
    image_ext: str,
    max_time_diff_camera_and_pose: float,
    save_overlay: bool,
    save_normal_map: bool,
):
    """Return camera params, transform, output subdirs, and synced image-pcd pairs."""
    K, D, h, w, fov_deg, _ = sensor.get_params_for_depth(cam_name, depth_pose_format, depth_pose_path)
    T_cam_base = sensor.tf.get_transform("base", cam_name)

    target_image_subdir = Path(image_folder_path) / subdir
    target_depth_subdir = output_depth_dir / subdir
    target_normal_subdir = output_normal_dir / subdir if save_normal_map else None
    target_overlay_subdir = overlay_dir / subdir if save_overlay else None

    logger.info(f"Target image directory: {target_image_subdir}")
    logger.info(f"Target depth directory: {target_depth_subdir}")

    image_pcd_pairs = get_image_pcd_sync_pair(
        target_image_subdir,
        slam_individual_clouds_new_path,
        image_ext,
        max_time_diff_camera_and_pose,
    )

    return (
        K,
        D,
        h,
        w,
        fov_deg,
        T_cam_base,
        image_pcd_pairs,
        target_depth_subdir,
        target_normal_subdir,
        target_overlay_subdir,
    )


def project_lidar_to_fisheye(
    sensor,
    project_dir: str,
    depth_dir: str,
    normal_dir: str,
    camera_topics_labelled: dict,
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
    proj_dir = Path(project_dir).expanduser()

    if is_euclidean:
        logger.info("Depth is euclidean: L2 distance between points and camera")
    else:
        logger.info("Depth is not euclidean: z_value")

    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"

    output_depth_dir, output_normal_dir, overlay_dir = setup_output_dirs(
        proj_dir, depth_dir, normal_dir, list(camera_topics_labelled.values()), save_overlay, save_normal_map
    )

    for cam_name, subdir in camera_topics_labelled.items():
        logger.info(f"Processing {cam_name} in {subdir} ...")
        K, D, h, w, fov_deg, T_cam_base, image_pcd_pairs, target_depth_subdir, target_normal_subdir, target_overlay_subdir = preprocess_camera(
            sensor, cam_name, subdir, image_folder_path, slam_individual_clouds_new_path,
            output_depth_dir, output_normal_dir, overlay_dir,
            depth_pose_format, depth_pose_path, image_ext, max_time_diff_camera_and_pose,
            save_overlay, save_normal_map,
        )  # fmt: skip

        logger.info(f"Fov: {fov_deg}")
        for image_path, pcd_path, _ in tqdm(image_pcd_pairs):
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if pcd is None:
                logger.warning(f"Failed to read {pcd_path}")
                continue
            if len(pcd.points) == 0:
                logger.warning(f"{pcd_path} is empty")
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


if __name__ == "__main__":
    config_yaml_path = Path(__file__).parent.parent / "configs" / "sensor.yaml"
    with open(config_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    sensor = Sensor(**yaml_data["sensor"])

    hf_repo_id = "ori-drs/oxford_spires_dataset"
    download_patterns = [
        "sequences/2024-03-18-christ-church-01/processed/vilens-slam/undist-clouds.zip",
        "sequences/2024-03-18-christ-church-01/processed/colmap/images.zip",
    ]
    local_dir = Path(__file__).parent.parent / "data" / "hf"
    for pattern in download_patterns:
        snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=pattern,
            local_dir=local_dir,
            repo_type="dataset",
            use_auth_token=False,
        )
    zip_files = list(Path(local_dir).rglob("*.zip"))
    unzip_files(zip_files)
    for zip_file in zip_files:
        zip_file.unlink()

    project_lidar_to_fisheye(
        sensor=sensor,
        project_dir=str(local_dir / "sequences/2024-03-18-christ-church-01/processed/oxspires_tools_outputs"),
        depth_dir="depths_euc_accum_0",
        normal_dir="normals_euc_accum_0",
        camera_topics_labelled=sensor.camera_topics_labelled,
        image_folder_path=str(local_dir / "sequences/2024-03-18-christ-church-01/processed/colmap"),
        depth_pose_format="vilens_slam",
        slam_individual_clouds_new_path=str(local_dir / "sequences/2024-03-18-christ-church-01/processed/vilens-slam/undist-clouds"),
        is_euclidean=True,
        max_time_diff_camera_and_pose=0.025,
        image_ext=".jpg",
        depth_factor=256.0,
        save_overlay=True,
        save_normal_map=True,
        camera_model="OPENCV_FISHEYE",
    )  # fmt: skip
