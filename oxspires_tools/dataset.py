import logging
from bisect import bisect_left
from pathlib import Path
from typing import List

from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp

logger = logging.getLogger(__name__)


def parse_lidar_timestamp(filename: str) -> str:
    assert filename.startswith("cloud_")
    # Remove "cloud_" prefix
    timestamp_str = filename[6:]
    # Replace last underscore with dot
    parts = timestamp_str.rsplit("_", 1)
    assert len(parts) == 2
    assert parts[0].isdigit()
    assert parts[1].isdigit()
    assert len(parts[0]) == 10  # sec should be 10 digits
    assert len(parts[1]) == 9  # nsec should be 9 digits
    return f"{parts[0]}.{parts[1]}"


def check_image_lidar_sync(
    image_dir: Path,
    lidar_dir: Path,
    tolerance_sec: float = 0.0,
) -> bool:
    """
    Check if LiDAR timestamps are synchronized with image timestamps.

    LiDAR has lower frequency than images, so we check if each LiDAR timestamp
    has a corresponding image timestamp (within tolerance).

    Args:
        image_dir: Directory containing image files (*.jpg)
        lidar_dir: Directory containing LiDAR point cloud files (*.pcd)
        tolerance_sec: Maximum allowed time difference in seconds.
            0.0 means exact match required.

    Returns:
        True if all LiDAR timestamps have matching image timestamps, False otherwise.
    """
    image_dir = Path(image_dir)
    lidar_dir = Path(lidar_dir)

    # Parse image timestamps
    image_timestamps: List[TimeStamp] = []
    for img_path in sorted(image_dir.glob("*.jpg")):
        ts = TimeStamp(t_string=img_path.stem)
        image_timestamps.append(ts)

    if not image_timestamps:
        logger.error(f"No images found in {image_dir}")
        return False

    # Parse LiDAR timestamps
    lidar_timestamps: List[TimeStamp] = []
    for pcd_path in sorted(lidar_dir.glob("*.pcd")):
        timestamp_str = parse_lidar_timestamp(pcd_path.stem)
        ts = TimeStamp(t_string=timestamp_str)
        lidar_timestamps.append(ts)

    if not lidar_timestamps:
        logger.error(f"No LiDAR point clouds found in {lidar_dir}")
        return False

    logger.info(f"Found {len(image_timestamps)} images, {len(lidar_timestamps)} LiDAR clouds")

    # Check synchronization
    unmatched: List[TimeStamp] = []
    if tolerance_sec == 0.0:
        # Exact match: use string comparison
        image_ts_set = {ts.t_string for ts in image_timestamps}
        for lidar_ts in lidar_timestamps:
            if lidar_ts.t_string not in image_ts_set:
                unmatched.append(lidar_ts)
    else:
        # Tolerance-based match: use float comparison with binary search
        image_floats = [ts.t_float128 for ts in image_timestamps]
        for lidar_ts in lidar_timestamps:
            lidar_float = lidar_ts.t_float128
            # Find closest image timestamp
            idx = bisect_left(image_floats, lidar_float)
            # Check both neighbors
            min_diff = float("inf")
            for i in [idx - 1, idx]:
                if 0 <= i < len(image_floats):
                    diff = abs(float(image_floats[i] - lidar_float))
                    min_diff = min(min_diff, diff)
            if min_diff > tolerance_sec:
                unmatched.append(lidar_ts)

    # Report results
    matched_count = len(lidar_timestamps) - len(unmatched)
    logger.info(f"Matched: {matched_count}/{len(lidar_timestamps)} LiDAR timestamps")

    if unmatched:
        logger.error(f"Unmatched LiDAR timestamps ({len(unmatched)}):")
        for ts in unmatched[:10]:  # Show first 10
            logger.error(f"  {ts.t_string}")
        if len(unmatched) > 10:
            logger.error(f"  ... and {len(unmatched) - 10} more")
        return False

    logger.info("All LiDAR timestamps are synchronized with images")
    return True
