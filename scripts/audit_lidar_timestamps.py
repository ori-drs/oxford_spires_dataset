"""Audit raw LiDAR scans for non-monotonic scan-to-scan point timestamps."""

import argparse
import json
from pathlib import Path

from oxspires_tools.lidar_undistortion.io import read_pcd_binary, scan_time_bounds_ns
from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp


def get_args():
    parser = argparse.ArgumentParser(
        description="Detect temporal overlap between consecutive raw LiDAR point clouds."
    )
    parser.add_argument("--lidar_dir", type=Path, required=True, help="Directory containing raw .pcd scans")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path for the audit report")
    parser.add_argument(
        "--fail_on_overlap",
        action="store_true",
        help="Exit with status 1 when at least one overlap is detected",
    )
    return parser.parse_args()


def _scan_start_from_filename_ns(path: Path) -> int:
    try:
        timestamp = TimeStamp(t_string=path.stem)
    except (AssertionError, ValueError) as exc:
        raise ValueError(f"Cannot parse scan timestamp from filename: {path.name}") from exc
    return timestamp.sec * 10**9 + timestamp.nsec


def audit_lidar_timestamps(lidar_dir: Path) -> dict:
    paths = sorted(lidar_dir.glob("*.pcd"), key=lambda path: path.stem)
    if not paths:
        raise ValueError(f"No PCD files found in {lidar_dir}")

    overlaps = []
    previous_path = None
    previous_end_ns = None
    durations_ns = []

    for path in paths:
        cloud, _ = read_pcd_binary(path)
        scan_start_ns = _scan_start_from_filename_ns(path) if "t" in cloud.dtype.names else None
        start_ns, end_ns = scan_time_bounds_ns(cloud, scan_start_ns=scan_start_ns)
        durations_ns.append(end_ns - start_ns)

        if previous_end_ns is not None and start_ns < previous_end_ns:
            overlap_ns = previous_end_ns - start_ns
            overlaps.append(
                {
                    "previous_scan": previous_path.name,
                    "current_scan": path.name,
                    "previous_end_ns": previous_end_ns,
                    "current_start_ns": start_ns,
                    "overlap_ns": overlap_ns,
                    "overlap_us": overlap_ns / 1e3,
                }
            )

        previous_path = path
        previous_end_ns = end_ns

    return {
        "lidar_dir": str(lidar_dir),
        "scan_count": len(paths),
        "overlap_count": len(overlaps),
        "max_overlap_ns": max((item["overlap_ns"] for item in overlaps), default=0),
        "mean_scan_duration_ms": sum(durations_ns) / len(durations_ns) / 1e6,
        "overlaps": overlaps,
    }


def main():
    args = get_args()
    report = audit_lidar_timestamps(args.lidar_dir)
    print(json.dumps(report, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    if args.fail_on_overlap and report["overlap_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
