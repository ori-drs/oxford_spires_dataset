"""Rectify (undistort) fisheye images using a sensor.yaml config."""

import argparse
import functools
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from oxspires_tools.sensor import Sensor
from oxspires_tools.utils import setup_logging

logger = logging.getLogger(__name__)

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def get_args():
    parser = argparse.ArgumentParser(description="Rectify fisheye images using sensor.yaml calibration")
    parser.add_argument("--cam_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0)  # fmt: skip
    parser.add_argument("--num_workers", type=int, default=28)  # fmt: skip
    return parser.parse_args()


def rectify_image(image_path, *, map1, map2, output_dir, cam_dir_name):
    """Rectify a single image and save to output_dir."""
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Failed to read {image_path}")
        return
    img_rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    out_path = output_dir / cam_dir_name / image_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_rect)


if __name__ == "__main__":
    setup_logging()
    args = get_args()

    default_config = Path(__file__).parent.parent / "configs" / "sensor.yaml"
    config_yaml_path = Path(args.config) if args.config else default_config
    with open(config_yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    sensor = Sensor(**yaml_data["sensor"])

    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    cam_names = list(sensor.camera_topics_labelled.keys())
    assert len(args.cam_dirs) == len(cam_names), f"Expected {len(cam_names)} cam_dirs, got {len(args.cam_dirs)}"

    for cam_name, cam_dir in zip(cam_names, args.cam_dirs):
        cam_dir = Path(cam_dir)
        logger.info(f"Rectifying {cam_name} ({cam_dir}) ...")

        K, D, h, w, _, _ = sensor.get_params_for_depth(cam_name, "vilens_slam", None)
        K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=args.balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_rect, (w, h), cv2.CV_16SC2)
        logger.info(f"  K_rect = {K_rect[0, 0]:.1f} {K_rect[1, 1]:.1f} {K_rect[0, 2]:.1f} {K_rect[1, 2]:.1f}")

        image_paths = sorted(p for p in cam_dir.iterdir() if p.suffix.lower() in _SUPPORTED_EXTS)
        logger.info(f"  {len(image_paths)} images")

        process = functools.partial(
            rectify_image, map1=map1, map2=map2, output_dir=output_dir, cam_dir_name=cam_dir.name
        )
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process, p) for p in image_paths]
            pbar = tqdm(total=len(futures), desc=cam_name)
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
            pbar.close()
