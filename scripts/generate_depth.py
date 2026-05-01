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


def get_args():
    parser = argparse.ArgumentParser(description="Generate depth maps from LiDAR point clouds")
    parser.add_argument("--cam_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--clouds_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_time_diff", type=float, default=None)
    parser.add_argument("--euclidean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth_factor", type=float, default=256.0)
    parser.add_argument("--accum_number", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = get_args()

    config_yaml_path = Path(__file__).parent.parent / "configs" / "sensor.yaml"
    with open(config_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    sensor = Sensor(**yaml_data["sensor"])

    max_time_diff = args.max_time_diff if args.max_time_diff is not None else sensor.max_time_diff_camera_and_pose
    depth_tag = "euc" if args.euclidean else "z"
    proj_dir = Path(args.output_dir)
    output_depth_dir = proj_dir / f"depths_{depth_tag}_accum_{args.accum_number}"
    output_normal_dir = proj_dir / f"normals_{depth_tag}_accum_{args.accum_number}"
    overlay_dir = proj_dir / f"depths_{depth_tag}_accum_{args.accum_number}_overlay"
    for d in (output_depth_dir, output_normal_dir, overlay_dir):
        shutil.rmtree(d, ignore_errors=True)

    cam_names = list(sensor.camera_topics_labelled.keys())
    assert len(args.cam_dirs) == len(cam_names), f"Expected exactly {len(cam_names)} cam_dirs, got {len(args.cam_dirs)}"
    logger.info("Depth is euclidean: L2 distance between points and camera" if args.euclidean else "Depth is z-value")
    for cam_name, cam_dir in zip(cam_names, args.cam_dirs):
        cam_dir = Path(cam_dir)
        logger.info(f"Processing {cam_name} ({cam_dir}) ...")
        K, D, h, w, fov_deg, _ = sensor.get_params_for_depth(cam_name, "vilens_slam", None)
        logger.info(f"Fov: {fov_deg}")
        T_cam_base = sensor.tf.get_transform("base", cam_name)
        image_pcd_pairs = get_image_pcd_sync_pair(cam_dir, Path(args.clouds_path), ".jpg", max_time_diff)  # fmt: skip

        for image_path, pcd_path, _ in tqdm(image_pcd_pairs):
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if pcd is None or len(pcd.points) == 0:
                logger.warning(f"Skipping {pcd_path}: {'failed to read' if pcd is None else 'empty'}")
                continue
            pcd.transform(T_cam_base)
            depth, normal = get_depth_from_cloud(
                pcd, K, D, w, h, fov_deg, "OPENCV_FISHEYE", args.depth_factor, args.euclidean
            )
            save_projection_outputs(
                depth,
                normal,
                image_path,
                save_depth_path=output_depth_dir / cam_dir.name / (image_path.stem + ".png"),
                save_normal_path=output_normal_dir / cam_dir.name / (image_path.stem + ".png"),
                save_overlay_path=overlay_dir / cam_dir.name / (image_path.stem + ".jpg"),
            )
