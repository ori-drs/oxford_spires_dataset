"""Evaluate a merged undistorted LiDAR cloud against a GT reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from oxspires_tools.eval import get_recon_metrics_multi_thresholds, save_error_cloud


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", type=Path, required=True, help="Merged undistorted cloud (.pcd or .ply)")  # fmt: skip
    parser.add_argument("--gt_cloud", type=Path, required=True, help="GT reconstruction cloud (.pcd or .ply)")  # fmt: skip
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for metrics and error cloud")  # fmt: skip
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling (m)")  # fmt: skip
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cloud: {args.cloud}")
    cloud = o3d.io.read_point_cloud(str(args.cloud))
    cloud_np = np.array(cloud.points)

    print(f"Loading GT cloud: {args.gt_cloud}")
    gt_cloud = o3d.io.read_point_cloud(str(args.gt_cloud))
    gt_np = np.array(gt_cloud.points)

    print(f"  cloud: {len(cloud_np)} points, GT: {len(gt_np)} points")

    metrics = get_recon_metrics_multi_thresholds(cloud_np, gt_np)

    base = metrics[0]
    print(f"\n  accuracy={base['accuracy'] * 100:.2f}cm  completeness={base['completeness'] * 100:.2f}cm")
    for m in metrics[1:]:
        print(f"    @{m['threshold']}m  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1_score']:.4f}")

    name = args.cloud.stem
    save_error_cloud(cloud_np, gt_np, str(args.output_dir / f"{name}_error.ply"))

    metrics_path = args.output_dir / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
