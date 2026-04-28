import argparse
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from oxspires_tools.utils import setup_logging

logger = logging.getLogger(__name__)

SEQ_DIR = "data/sequences"

# Expected camera subdirs inside raw/
RAW_CAMS = ["cam0", "cam1", "cam2"]

# Expected trajectory files
TRAJ_FILES = ["gt-tum.txt", "vilens-slam-tum.txt", "hba-tum.txt", "colmap-tum.txt"]

# Expected ROS topics and their corresponding raw data
BAG_TOPIC_TO_RAW = {
    "/alphasense_driver_ros/cam0/debayered/image/compressed": "raw/cam0",
    "/alphasense_driver_ros/cam1/debayered/image/compressed": "raw/cam1",
    "/alphasense_driver_ros/cam2/debayered/image/compressed": "raw/cam2",
    "/alphasense_driver_ros/imu": "raw/imu.csv",
    "/hesai/pandar": "raw/lidar-clouds",
}

# COLMAP image subdirs (after images.zip is unpacked)
COLMAP_CAM_DIRS = [
    "alphasense_driver_ros_cam0_debayered_image_compressed",
    "alphasense_driver_ros_cam1_debayered_image_compressed",
    "alphasense_driver_ros_cam2_debayered_image_compressed",
]


@dataclass
class CheckResult:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def ok(self, msg: str):
        self.passed.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def error(self, msg: str):
        self.errors.append(msg)
        logger.error(msg)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def _parse_image_timestamps(cam_dir: Path) -> list[float]:
    """Parse float timestamps from {sec}.{nsec}.jpg filenames."""
    ts = []
    for p in sorted(cam_dir.glob("*.jpg")):
        try:
            ts.append(float(p.stem))
        except ValueError:
            pass
    return ts


def _parse_tum(path: Path) -> np.ndarray:
    """Load TUM file, skip comment lines, return Nx8 float array."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    return np.array(rows) if rows else np.empty((0, 8))


def _check_quaternions(poses: np.ndarray, label: str, result: CheckResult):
    """Check columns 4:8 are unit quaternions."""
    if poses.shape[0] == 0:
        return
    quats = poses[:, 4:8]
    norms = np.linalg.norm(quats, axis=1)
    bad = np.where(np.abs(norms - 1.0) > 1e-3)[0]
    if len(bad):
        result.error(f"{label}: {len(bad)} non-unit quaternions (e.g. row {bad[0]}, norm={norms[bad[0]]:.4f})")
    else:
        result.ok(f"{label}: all quaternions unit-norm")


def check_file_existence(seq_dir: Path, result: CheckResult):
    """Check all required files/dirs exist."""
    required = (
        [seq_dir / "raw" / cam for cam in RAW_CAMS]
        + [
            seq_dir / "raw" / "imu.csv",
            seq_dir / "processed" / "vilens-slam" / "undist-clouds",
            seq_dir / "processed" / "vilens-slam" / "slam-poses.csv",
        ]
        + [seq_dir / "processed" / "trajectory" / f for f in TRAJ_FILES]
        + [
            seq_dir / "processed" / "colmap" / "database.db",
            seq_dir / "processed" / "colmap" / "0" / "cameras.bin",
            seq_dir / "processed" / "colmap" / "0" / "images.bin",
            seq_dir / "processed" / "colmap" / "0" / "points3D.bin",
            seq_dir / "processed" / "colmap" / "transforms_colmap.json",
            seq_dir / "processed" / "colmap" / "transforms_colmap_scaled.json",
            seq_dir / "processed" / "colmap" / "evo_align_results.json",
        ]
    )
    ros_bag_dir = seq_dir / "raw" / "ros2bag"
    rosbag_dir = seq_dir / "raw" / "rosbag"
    for p in required:
        if not p.exists():
            result.error(f"Missing: {p.relative_to(seq_dir)}")
        else:
            result.ok(f"Exists: {p.relative_to(seq_dir)}")
    # ROS bags: just check dirs exist and are non-empty
    for bag_dir, ext in [(ros_bag_dir, ".db3"), (rosbag_dir, ".bag")]:
        if not bag_dir.exists():
            result.error(f"Missing dir: {bag_dir.relative_to(seq_dir)}")
        else:
            files = list(bag_dir.rglob(f"*{ext}"))
            if not files:
                result.error(f"No {ext} files in {bag_dir.relative_to(seq_dir)}")
            else:
                for f in files:
                    size_mb = f.stat().st_size / 1e6
                    if size_mb < 1.0:
                        result.warn(f"Suspiciously small bag ({size_mb:.1f} MB): {f.relative_to(seq_dir)}")
                    else:
                        result.ok(f"Bag exists ({size_mb:.0f} MB): {f.relative_to(seq_dir)}")


def check_raw_images(seq_dir: Path, result: CheckResult):
    """Check raw camera images."""
    all_counts = {}
    all_ts = {}
    for cam in RAW_CAMS:
        cam_dir = seq_dir / "raw" / cam
        if not cam_dir.exists():
            result.error(f"raw/{cam}: directory missing, skipping image checks")
            continue
        jpgs = sorted(cam_dir.glob("*.jpg"))
        if not jpgs:
            result.error(f"raw/{cam}: no images found")
            continue

        # Filename format
        bad_names = [p for p in jpgs if len(p.stem.split(".")) != 2]
        if bad_names:
            result.warn(f"raw/{cam}: {len(bad_names)} files with unexpected name format")

        # Zero-byte check
        empty = [p for p in jpgs if p.stat().st_size == 0]
        if empty:
            result.error(f"raw/{cam}: {len(empty)} zero-byte images")

        ts = _parse_image_timestamps(cam_dir)
        all_counts[cam] = len(ts)
        all_ts[cam] = ts

        # Monotonicity
        diffs = np.diff(ts)
        if np.any(diffs <= 0):
            result.error(f"raw/{cam}: timestamps not strictly increasing")
        else:
            result.ok(f"raw/{cam}: timestamps strictly increasing")

        # Frequency
        duration = ts[-1] - ts[0]
        fps = len(ts) / duration if duration > 0 else 0
        if not (18 <= fps <= 22):
            result.warn(f"raw/{cam}: frame rate {fps:.1f} Hz (expected ~20 Hz)")
        else:
            result.ok(f"raw/{cam}: {len(ts)} images, {fps:.1f} Hz, {duration:.1f}s")

    # Count consistency across cameras
    if len(all_counts) == len(RAW_CAMS):
        counts = list(all_counts.values())
        max_c, min_c = max(counts), min(counts)
        if max_c - min_c > max_c * 0.01:
            result.warn(f"Camera image counts diverge: { {k: v for k, v in all_counts.items()} }")
        else:
            result.ok(f"Camera image counts consistent: { {k: v for k, v in all_counts.items()} }")

        # Timestamp cross-camera sync
        ref_set = set(f"{t:.9f}" for t in all_ts.get("cam0", []))
        for cam in ["cam1", "cam2"]:
            if cam not in all_ts:
                continue
            other_set = set(f"{t:.9f}" for t in all_ts[cam])
            missing = ref_set - other_set
            extra = other_set - ref_set
            if missing or extra:
                result.warn(f"raw/{cam} vs cam0: {len(missing)} missing, {len(extra)} extra timestamps")
            else:
                result.ok(f"raw/{cam}: timestamps match cam0")


def check_imu(seq_dir: Path, result: CheckResult):
    """Check IMU CSV."""
    imu_path = seq_dir / "raw" / "imu.csv"
    if not imu_path.exists():
        result.error("raw/imu.csv: missing, skipping IMU checks")
        return

    import csv

    expected_cols = ["secs", "nsecs", "acc_x", "acc_y", "acc_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
    rows = []
    with open(imu_path) as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames) != expected_cols:
            result.error(f"raw/imu.csv: unexpected columns {reader.fieldnames}")
            return
        for row in reader:
            rows.append(row)

    if not rows:
        result.error("raw/imu.csv: empty")
        return

    # Build float timestamps
    ts = np.array([int(r["secs"]) + int(r["nsecs"]) * 1e-9 for r in rows])

    # Monotonicity
    if np.any(np.diff(ts) <= 0):
        result.error("raw/imu.csv: timestamps not strictly increasing")
    else:
        result.ok("raw/imu.csv: timestamps strictly increasing")

    # Frequency
    duration = ts[-1] - ts[0]
    rate = len(ts) / duration if duration > 0 else 0
    if not (350 <= rate <= 450):
        result.warn(f"raw/imu.csv: rate {rate:.1f} Hz (expected ~400 Hz)")
    else:
        result.ok(f"raw/imu.csv: {len(ts)} rows, {rate:.1f} Hz")

    # NaN/inf
    data_cols = ["acc_x", "acc_y", "acc_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
    for col in data_cols:
        vals = np.array([float(r[col]) for r in rows])
        if not np.all(np.isfinite(vals)):
            result.error(f"raw/imu.csv: NaN/inf in column {col}")


def check_undist_clouds(seq_dir: Path, result: CheckResult):
    """Check undistorted LiDAR point clouds."""
    cloud_dir = seq_dir / "processed" / "vilens-slam" / "undist-clouds"
    if not cloud_dir.exists():
        result.error("processed/vilens-slam/undist-clouds: missing, skipping")
        return

    pcds = sorted(cloud_dir.glob("*.pcd"))
    if not pcds:
        result.error("processed/vilens-slam/undist-clouds: no PCD files found")
        return

    # Filename format: cloud_{sec}_{nsec}.pcd
    bad = [p for p in pcds if not p.stem.startswith("cloud_") or len(p.stem.split("_")) != 3]
    if bad:
        result.error(f"undist-clouds: {len(bad)} files with unexpected name format")

    # Zero-byte
    empty = [p for p in pcds if p.stat().st_size == 0]
    if empty:
        result.error(f"undist-clouds: {len(empty)} zero-byte PCD files")

    # Timestamps
    ts = []
    for p in pcds:
        parts = p.stem.split("_")
        try:
            ts.append(float(f"{parts[1]}.{parts[2]}"))
        except (ValueError, IndexError):
            pass

    if len(ts) < 2:
        result.warn("undist-clouds: too few timestamps to check ordering/rate")
        return

    ts_arr = np.array(ts)
    if np.any(np.diff(ts_arr) <= 0):
        result.error("undist-clouds: timestamps not strictly increasing")
    else:
        result.ok("undist-clouds: timestamps strictly increasing")

    duration = ts_arr[-1] - ts_arr[0]
    rate = len(ts) / duration if duration > 0 else 0
    result.ok(f"undist-clouds: {len(pcds)} files, {rate:.2f} Hz, {duration:.1f}s")

    # Sync with cam0 (clouds outside image window are expected — warn only)
    cam0_dir = seq_dir / "raw" / "cam0"
    if cam0_dir.exists():
        cam0_ts = set()
        cam0_floats = []
        for p in sorted(cam0_dir.glob("*.jpg")):
            cam0_ts.add(p.stem)
            cam0_floats.append(float(p.stem))
        img_t_min = min(cam0_floats) if cam0_floats else None
        img_t_max = max(cam0_floats) if cam0_floats else None
        unmatched_inside = []
        unmatched_outside = []
        for p in pcds:
            parts = p.stem.split("_")
            img_stem = f"{parts[1]}.{parts[2]}"
            cloud_t = float(img_stem)
            if img_stem not in cam0_ts:
                if img_t_min is not None and img_t_min <= cloud_t <= img_t_max:
                    unmatched_inside.append(img_stem)
                else:
                    unmatched_outside.append(img_stem)
        if unmatched_inside:
            # Check if they share the nsec-truncated-to-000 pattern (old file format artifact)
            truncated = [s for s in unmatched_inside if s.endswith("000")]
            if len(truncated) == len(unmatched_inside):
                result.error(
                    f"undist-clouds: {len(unmatched_inside)} clouds within image window have no matching cam0 image"
                    f" — all have nsec truncated to microseconds (old file format, likely stale files mixed with new)"
                )
            else:
                result.error(
                    f"undist-clouds: {len(unmatched_inside)} clouds within image window have no matching cam0 image"
                    f" (e.g. {unmatched_inside[0]})"
                )
        else:
            result.ok("undist-clouds: all clouds within image window match cam0 images")
        if unmatched_outside:
            result.warn(f"undist-clouds: {len(unmatched_outside)} clouds outside image time window (expected)")

    return len(pcds)


def check_slam_poses(seq_dir: Path, n_clouds: int | None, result: CheckResult):
    """Check slam-poses.csv."""
    poses_path = seq_dir / "processed" / "vilens-slam" / "slam-poses.csv"
    if not poses_path.exists():
        result.error("slam-poses.csv: missing, skipping")
        return

    rows = []
    with open(poses_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(","))

    if not rows:
        result.error("slam-poses.csv: empty")
        return

    try:
        data = np.array([[float(v) for v in r] for r in rows])
    except ValueError as e:
        result.error(f"slam-poses.csv: parse error: {e}")
        return

    result.ok(f"slam-poses.csv: {len(data)} poses")

    if n_clouds is not None and abs(len(data) - n_clouds) > 1:
        result.warn(
            f"slam-poses.csv: {len(data)} poses vs {n_clouds} PCD files (mismatch — may indicate cloud set was updated)"
        )

    if not np.all(np.isfinite(data)):
        result.error("slam-poses.csv: NaN/inf values found")

    # Quaternion norm (cols 6:10 = qx qy qz qw)
    quats = data[:, 6:10]
    norms = np.linalg.norm(quats, axis=1)
    bad = np.where(np.abs(norms - 1.0) > 1e-3)[0]
    if len(bad):
        result.error(f"slam-poses.csv: {len(bad)} non-unit quaternions")
    else:
        result.ok("slam-poses.csv: all quaternions unit-norm")


def check_trajectories(seq_dir: Path, result: CheckResult):
    """Check all trajectory TUM files."""
    traj_dir = seq_dir / "processed" / "trajectory"
    # Get image time range for coverage check
    cam0_dir = seq_dir / "raw" / "cam0"
    img_ts = _parse_image_timestamps(cam0_dir) if cam0_dir.exists() else []
    t_min = min(img_ts) - 5.0 if img_ts else None
    t_max = max(img_ts) + 5.0 if img_ts else None

    for fname in TRAJ_FILES:
        path = traj_dir / fname
        if not path.exists():
            result.error(f"trajectory/{fname}: missing")
            continue

        poses = _parse_tum(path)
        if poses.shape[0] == 0:
            result.error(f"trajectory/{fname}: empty")
            continue
        if poses.shape[1] != 8:
            result.error(f"trajectory/{fname}: expected 8 columns, got {poses.shape[1]}")
            continue

        result.ok(f"trajectory/{fname}: {len(poses)} poses")

        if not np.all(np.isfinite(poses)):
            result.error(f"trajectory/{fname}: NaN/inf values")
            continue

        # Monotonicity
        diffs = np.diff(poses[:, 0])
        if np.any(diffs <= 0):
            result.error(f"trajectory/{fname}: timestamps not strictly increasing")
        else:
            result.ok(f"trajectory/{fname}: timestamps strictly increasing")

        # Timestamp range: must overlap with images, but extending beyond is expected
        if t_min is not None:
            t_start, t_end = poses[0, 0], poses[-1, 0]
            overlap = t_start <= t_max and t_end >= t_min
            if not overlap:
                result.error(
                    f"trajectory/{fname}: no overlap with image range [{t_min:.1f}, {t_max:.1f}]"
                    f" (traj=[{t_start:.1f}, {t_end:.1f}])"
                )
            else:
                result.ok(
                    f"trajectory/{fname}: overlaps image range"
                    f" (traj=[{t_start:.1f}, {t_end:.1f}], images=[{t_min:.1f}, {t_max:.1f}])"
                )

        _check_quaternions(poses, f"trajectory/{fname}", result)


def check_colmap(seq_dir: Path, result: CheckResult):
    """Check COLMAP outputs."""
    colmap_dir = seq_dir / "processed" / "colmap"
    if not colmap_dir.exists():
        result.error("processed/colmap: missing, skipping")
        return

    # database.db
    db_path = colmap_dir / "database.db"
    if db_path.exists() and db_path.stat().st_size > 0:
        try:
            conn = sqlite3.connect(db_path)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            conn.close()
            result.ok(f"colmap/database.db: valid SQLite, tables={tables}")
        except Exception as e:
            result.error(f"colmap/database.db: SQLite error: {e}")
    else:
        result.error("colmap/database.db: missing or empty")

    # Sparse model files
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = colmap_dir / "0" / fname
        if not p.exists() or p.stat().st_size == 0:
            result.error(f"colmap/0/{fname}: missing or empty")
        else:
            result.ok(f"colmap/0/{fname}: {p.stat().st_size / 1e3:.0f} KB")

    # transforms_colmap.json
    import json

    for json_name in ["transforms_colmap.json", "transforms_colmap_scaled.json"]:
        json_path = colmap_dir / json_name
        if not json_path.exists():
            result.error(f"colmap/{json_name}: missing")
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            result.error(f"colmap/{json_name}: JSON parse error: {e}")
            continue

        if "camera_model" not in data or "frames" not in data:
            result.error(f"colmap/{json_name}: missing 'camera_model' or 'frames' keys")
            continue

        result.ok(f"colmap/{json_name}: camera_model={data['camera_model']}, {len(data['frames'])} frames")

        # Check frame fields and transform matrices
        bad_frames = 0
        missing_files = 0
        for fr in data["frames"]:
            mat = fr.get("transform_matrix", [])
            if len(mat) != 4 or any(len(row) != 4 for row in mat):
                bad_frames += 1
                continue
            flat = [v for row in mat for v in row]
            if not all(math.isfinite(v) for v in flat):
                bad_frames += 1
            # Check image file existence (strip leading "images/" prefix if present)
            fp = fr.get("file_path", "")
            if fp.startswith("images/"):
                fp = fp[len("images/") :]
            img_path = colmap_dir / fp
            if not img_path.exists():
                missing_files += 1

        if bad_frames:
            result.error(f"colmap/{json_name}: {bad_frames} frames with invalid transform_matrix")
        else:
            result.ok(f"colmap/{json_name}: all transform_matrices valid (4x4, finite)")
        if missing_files:
            result.error(f"colmap/{json_name}: {missing_files} file_path entries point to missing images")
        else:
            result.ok(f"colmap/{json_name}: all referenced images exist")

    # evo_align_results.json
    evo_path = colmap_dir / "evo_align_results.json"
    if evo_path.exists():
        try:
            with open(evo_path) as f:
                data = json.load(f)
            if "rotation" not in data or "translation" not in data:
                result.error("colmap/evo_align_results.json: missing 'rotation' or 'translation' keys")
            else:
                flat = [v for row in data["rotation"] for v in row] + list(data["translation"])
                if not all(math.isfinite(v) for v in flat):
                    result.error("colmap/evo_align_results.json: non-finite values")
                else:
                    result.ok("colmap/evo_align_results.json: valid")
        except Exception as e:
            result.error(f"colmap/evo_align_results.json: parse error: {e}")

    # COLMAP image subdirs
    for cam_subdir in COLMAP_CAM_DIRS:
        d = colmap_dir / cam_subdir
        if not d.exists():
            result.error(f"colmap/{cam_subdir}: missing")
        else:
            n = len(list(d.glob("*.jpg")))
            if n == 0:
                result.error(f"colmap/{cam_subdir}: no images")
            else:
                result.ok(f"colmap/{cam_subdir}: {n} images")


def _read_bag_topic_counts(bag_path_or_dir: Path, is_ros2: bool) -> dict[str, int] | None:
    """Return {topic: message_count} from a ROS1 or ROS2 bag. Returns None on import error."""
    try:
        if is_ros2:
            from rosbags.rosbag2 import Reader
        else:
            from rosbags.rosbag1 import Reader
    except ImportError:
        return None
    with Reader(bag_path_or_dir) as r:
        return {topic: info.msgcount for topic, info in r.topics.items()}


def _read_bag_timestamp_strings(bag_path_or_dir: Path, topic: str, is_ros2: bool) -> list[str] | None:
    """
    Return sorted list of header timestamp strings ("{sec}.{nsec:09d}") for a topic.
    Parses only the first 12 bytes of each CDR message to extract stamp.sec + stamp.nanosec
    without full deserialization — works for any message whose header stamp is first.
    Uses integer sec/nsec to avoid float precision loss.
    Returns None on import error.
    """
    try:
        if is_ros2:
            from rosbags.rosbag2 import Reader
        else:
            from rosbags.rosbag1 import Reader
    except ImportError:
        return None

    from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp

    timestamps = []
    with Reader(bag_path_or_dir) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            return []
        for _, _, raw in r.messages(connections=conns):
            # CDR layout: 4-byte header, then stamp.sec (int32) + stamp.nanosec (uint32)
            sec, nsec = struct.unpack_from("<iI", raw, offset=4)
            timestamps.append(TimeStamp(sec=sec, nsec=nsec).t_string)
    return sorted(timestamps)


def _bag_window(bag_path_or_dir: Path, is_ros2: bool) -> tuple[float, float] | None:
    """Return (start_time_s, end_time_s) of the bag recording window."""
    try:
        if is_ros2:
            from rosbags.rosbag2 import Reader
        else:
            from rosbags.rosbag1 import Reader
    except ImportError:
        return None
    with Reader(bag_path_or_dir) as r:
        return r.start_time * 1e-9, (r.start_time + r.duration) * 1e-9


def check_rosbags(seq_dir: Path, result: CheckResult):
    """Check ROS1 and ROS2 bag consistency against raw data."""
    raw_dir = seq_dir / "raw"

    # Locate bags
    ros2_dirs = sorted((raw_dir / "ros2bag").glob("*/")) if (raw_dir / "ros2bag").exists() else []
    ros1_bags = sorted((raw_dir / "rosbag").glob("*.bag")) if (raw_dir / "rosbag").exists() else []

    if not ros2_dirs:
        result.error("rosbags: no ros2bag directory found, skipping bag checks")
        return
    if not ros1_bags:
        result.error("rosbags: no ros1 .bag file found, skipping bag checks")
        return

    ros2_dir = ros2_dirs[0]
    ros1_bag = ros1_bags[0]

    # --- Message count consistency between ROS1 and ROS2 ---
    ros2_counts = _read_bag_topic_counts(ros2_dir, is_ros2=True)
    ros1_counts = _read_bag_topic_counts(ros1_bag, is_ros2=False)

    if ros2_counts is None or ros1_counts is None:
        result.warn("rosbags: rosbags library not available, skipping bag checks")
        return

    for topic in sorted(BAG_TOPIC_TO_RAW):
        c2 = ros2_counts.get(topic)
        c1 = ros1_counts.get(topic)
        if c2 is None:
            result.error(f"ros2bag: missing topic {topic}")
        if c1 is None:
            result.error(f"ros1bag: missing topic {topic}")
        if c1 is not None and c2 is not None:
            if c1 != c2:
                result.error(f"ros1/ros2 count mismatch on {topic}: ros1={c1} ros2={c2}")
            else:
                result.ok(f"ros1==ros2 count for {topic}: {c1}")

    # --- Bag window ---
    window = _bag_window(ros2_dir, is_ros2=True)
    if window is None:
        return
    bag_t_start, bag_t_end = window

    # --- Camera: count and timestamp match ---
    for cam_idx, topic in enumerate(
        [
            "/alphasense_driver_ros/cam0/debayered/image/compressed",
            "/alphasense_driver_ros/cam1/debayered/image/compressed",
            "/alphasense_driver_ros/cam2/debayered/image/compressed",
        ]
    ):
        cam = f"cam{cam_idx}"
        cam_dir = raw_dir / cam
        if not cam_dir.exists():
            result.error(f"rosbags/{cam}: raw dir missing, skipping")
            continue

        raw_stems = sorted(p.stem for p in cam_dir.glob("*.jpg"))
        bag_count = ros2_counts.get(topic, 0)

        # Count check
        if len(raw_stems) != bag_count:
            result.error(f"rosbags/{cam}: raw file count ({len(raw_stems)}) != bag message count ({bag_count})")
        else:
            result.ok(f"rosbags/{cam}: count matches bag ({bag_count})")

        # Timestamp check — compare header timestamps from bag vs image filenames
        bag_stems = _read_bag_timestamp_strings(ros2_dir, topic, is_ros2=True)
        if bag_stems is None:
            continue

        mismatched = [(b, r) for b, r in zip(bag_stems, raw_stems) if b != r]
        if mismatched:
            result.error(
                f"rosbags/{cam}: {len(mismatched)} header timestamps don't match filenames"
                f" (e.g. bag={mismatched[0][0]} file={mismatched[0][1]})"
            )
        else:
            result.ok(f"rosbags/{cam}: all header timestamps match image filenames")

    # --- IMU: count and time range ---
    imu_path = raw_dir / "imu.csv"
    bag_imu_count = ros2_counts.get("/alphasense_driver_ros/imu", 0)
    if imu_path.exists():
        import csv

        with open(imu_path) as f:
            imu_rows = sum(1 for _ in csv.reader(f)) - 1  # subtract header
        if imu_rows != bag_imu_count:
            result.error(f"rosbags/imu: imu.csv rows ({imu_rows}) != bag message count ({bag_imu_count})")
        else:
            result.ok(f"rosbags/imu: count matches bag ({bag_imu_count})")

        # Timestamp range: imu.csv may start slightly before bag window (expected)
        bag_imu_strings = _read_bag_timestamp_strings(ros2_dir, "/alphasense_driver_ros/imu", is_ros2=True)
        if bag_imu_strings:
            with open(imu_path) as f:
                rows = list(csv.DictReader(f))
            from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp

            csv_first = TimeStamp(sec=int(rows[0]["secs"]), nsec=int(rows[0]["nsecs"]))
            csv_last = TimeStamp(sec=int(rows[-1]["secs"]), nsec=int(rows[-1]["nsecs"]))
            bag_first = TimeStamp(t_string=bag_imu_strings[0])
            bag_last = TimeStamp(t_string=bag_imu_strings[-1])
            # Compare using float128 for safe arithmetic
            dt_first = abs(float(csv_first.t_float128 - bag_first.t_float128))
            dt_last = abs(float(csv_last.t_float128 - bag_last.t_float128))
            if dt_first > 0.1 or dt_last > 0.1:
                result.error(
                    f"rosbags/imu: timestamp range mismatch — "
                    f"csv=[{csv_first.t_string},{csv_last.t_string}] "
                    f"bag=[{bag_first.t_string},{bag_last.t_string}]"
                )
            else:
                result.ok(
                    f"rosbags/imu: time range matches "
                    f"(start delta={dt_first * 1e3:.1f}ms, end delta={dt_last * 1e3:.1f}ms)"
                )

    # --- LiDAR: count of raw clouds inside bag window ---
    lidar_dir = raw_dir / "lidar-clouds"
    bag_lidar_count = ros2_counts.get("/hesai/pandar", 0)
    if lidar_dir.exists():
        lidar_files = sorted(lidar_dir.glob("*.pcd"))
        # Use float comparison for window check — 1ms precision is sufficient here
        lidar_inside_window = [p for p in lidar_files if bag_t_start - 0.001 <= float(p.stem) <= bag_t_end + 0.001]
        if len(lidar_inside_window) != bag_lidar_count:
            result.error(
                f"rosbags/lidar: raw clouds inside bag window ({len(lidar_inside_window)}) "
                f"!= bag message count ({bag_lidar_count})"
            )
        else:
            result.ok(
                f"rosbags/lidar: raw cloud count inside bag window matches ({bag_lidar_count}); "
                f"{len(lidar_files) - len(lidar_inside_window)} clouds outside window (expected)"
            )


def check_sequence(seq_dir: Path) -> CheckResult:
    """Run all checks for a single sequence."""
    result = CheckResult()

    logger.info("--- File existence ---")
    check_file_existence(seq_dir, result)

    logger.info("--- Raw images ---")
    check_raw_images(seq_dir, result)

    logger.info("--- IMU ---")
    check_imu(seq_dir, result)

    logger.info("--- Undistorted clouds ---")
    n_clouds = check_undist_clouds(seq_dir, result)

    logger.info("--- SLAM poses ---")
    check_slam_poses(seq_dir, n_clouds, result)

    logger.info("--- Trajectories ---")
    check_trajectories(seq_dir, result)

    logger.info("--- COLMAP ---")
    check_colmap(seq_dir, result)

    logger.info("--- ROS bags ---")
    check_rosbags(seq_dir, result)

    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--sequence", type=str, default=None, help="Check a single sequence by name")
    return parser.parse_args()


def main():
    setup_logging()
    args = get_args()

    sequences_dir = Path(args.data_dir) / "sequences"
    if args.sequence:
        seq_dirs = [sequences_dir / args.sequence]
    else:
        seq_dirs = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])

    summary = {}
    pbar = tqdm(total=len(seq_dirs))
    for seq_dir in seq_dirs:
        pbar.set_description(seq_dir.name)
        logger.info(f"\n{'=' * 60}\nChecking: {seq_dir.name}\n{'=' * 60}")
        result = check_sequence(seq_dir)
        summary[seq_dir.name] = result
        status = "PASS" if result.success else "FAIL"
        logger.info(
            f"{status}: {len(result.passed)} passed, {len(result.warnings)} warnings, {len(result.errors)} errors"
        )
        pbar.update(1)
    pbar.close()

    # Final summary
    logger.info(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    all_pass = True
    for name, result in summary.items():
        status = "PASS" if result.success else "FAIL"
        if not result.success:
            all_pass = False
        logger.info(f"  {status}  {name}  (errors={len(result.errors)}, warnings={len(result.warnings)})")
        for err in result.errors:
            logger.error(f"        ERROR: {err}")

    if all_pass:
        logger.info("\nAll sequences passed!")
    else:
        logger.error("\nSome sequences have errors — see above.")


if __name__ == "__main__":
    main()
