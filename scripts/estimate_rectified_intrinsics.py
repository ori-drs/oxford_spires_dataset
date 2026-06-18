"""Estimate rectified (pinhole) intrinsics from a fisheye sensor.yaml and write sensor_rect.yaml."""

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0)  # fmt: skip
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    default_config = Path(__file__).parent.parent / "configs" / "sensor.yaml"
    config_path = Path(args.config) if args.config else default_config
    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    assert yaml_data["sensor"]["camera_model"] == "OPENCV_FISHEYE", "Input must be OPENCV_FISHEYE"

    rect_yaml = copy.deepcopy(yaml_data)
    rect_yaml["sensor"]["camera_model"] = "OPENCV"

    max_fov = 0.0
    for cam in rect_yaml["sensor"]["cameras"]:
        fx, fy, cx, cy = cam["intrinsics"]
        k1, k2, k3, k4 = cam["extra_params"]
        h, w = cam["image_height"], cam["image_width"]

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([k1, k2, k3, k4], dtype=np.float64)
        K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=args.balance)

        fx_r, fy_r, cx_r, cy_r = K_rect[0, 0], K_rect[1, 1], K_rect[0, 2], K_rect[1, 2]
        fov_deg = 2 * np.degrees(np.arctan(np.sqrt((w / 2 / fx_r) ** 2 + (h / 2 / fy_r) ** 2)))
        max_fov = max(max_fov, fov_deg)

        cam["intrinsics"] = [float(fx_r), float(fy_r), float(cx_r), float(cy_r)]
        cam["extra_params"] = [0.0, 0.0, 0.0, 0.0]
        print(f"{cam['label']}: K_rect = {cam['intrinsics']}, fov = {fov_deg:.1f} deg")

    rect_yaml["sensor"]["camera_fov"] = round(max_fov + 2, 1)

    output_path = Path(args.output) if args.output else config_path.parent / "sensor_rect.yaml"
    with open(output_path, "w") as f:
        yaml.dump(rect_yaml, f, default_flow_style=None, sort_keys=False)
    print(f"Written to {output_path}")
