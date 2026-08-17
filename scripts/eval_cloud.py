"""Evaluate a single reconstructed cloud against a GT cloud and print/save metrics."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from oxspires_tools.eval import get_recon_metrics
from oxspires_tools.utils import setup_logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_cloud", type=Path, required=True, help="Input cloud (.pcd or .ply)")  # fmt: skip
    parser.add_argument("--gt_cloud", type=Path, required=True, help="GT cloud (.pcd or .ply)")  # fmt: skip
    parser.add_argument("--output_json", type=Path, default=None, help="Output JSON for metrics")  # fmt: skip
    parser.add_argument("--output_error_cloud_dir", type=Path, default=None, help="Directory to save error cloud; defaults to <output_json parent>/<stem>_error/ if --output_json is set")  # fmt: skip
    parser.add_argument("--precision_threshold", type=float, default=0.05, help="Input-to-GT precision threshold in metres")  # fmt: skip
    parser.add_argument("--recall_threshold", type=float, default=0.05, help="GT-to-input recall threshold in metres")  # fmt: skip
    parser.add_argument("--max_distance", type=float, default=np.inf, help="Maximum KD-tree correspondence distance in metres")  # fmt: skip
    parser.add_argument("--one_sided", action="store_true", help="Skip GT-to-input completeness/recall computation")  # fmt: skip
    return parser.parse_args()


def main():
    setup_logging()
    args = get_args()
    logging.info(f"Loading input: {args.input_cloud}")
    input_np = np.asarray(o3d.io.read_point_cloud(str(args.input_cloud)).points)
    logging.info(f"Loading GT:    {args.gt_cloud}")
    gt_np = np.asarray(o3d.io.read_point_cloud(str(args.gt_cloud)).points)
    logging.info(f"  Input: {len(input_np)} points, GT: {len(gt_np)} points")

    error_cloud_dir = args.output_error_cloud_dir
    if error_cloud_dir is None and args.output_json is not None:
        error_cloud_dir = args.output_json.parent / (args.output_json.stem + "_error")
    if error_cloud_dir is not None:
        error_cloud_dir.mkdir(parents=True, exist_ok=True)

    results = get_recon_metrics(
        input_np,
        gt_np,
        precision_threshold=args.precision_threshold,
        recall_threshold=args.recall_threshold,
        compute_precision=True,
        compute_recall=not args.one_sided,
        save_error_cloud_dir=error_cloud_dir,
        max_distance=args.max_distance,
    )
    logging.info(f"Metrics for {args.input_cloud.stem}:\n{json.dumps(results, indent=2)}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
