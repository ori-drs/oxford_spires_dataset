"""Merge and downsample a directory of PCD files into a single cloud."""

import argparse
from pathlib import Path

from oxspires_tools.point_cloud import merge_downsample_clouds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clouds_dir", type=Path, required=True, help="Directory of PCD files to merge")  # fmt: skip
    parser.add_argument("--output_pcd", type=Path, required=True, help="Output merged PCD path")  # fmt: skip
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling (m)")  # fmt: skip
    return parser.parse_args()


def main():
    args = get_args()
    cloud_paths = sorted(args.clouds_dir.glob("*.pcd"))
    if not cloud_paths:
        raise FileNotFoundError(f"No PCD files found in {args.clouds_dir}")
    args.output_pcd.parent.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(cloud_paths)} clouds -> {args.output_pcd}")
    merge_downsample_clouds(cloud_paths, args.output_pcd, args.voxel_size)


if __name__ == "__main__":
    main()
