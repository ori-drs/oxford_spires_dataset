import logging
from pathlib import Path
from typing import List

from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp
from oxspires_tools.utils import find_closest_in_sorted

logger = logging.getLogger(__name__)


class OxfordSpiresDataset:
    """Centralised path management and common checks for a SPIRES sequence."""

    RAW_CAM_IDS = (0, 1, 2)
    TRAJECTORY_FILES = ("gt-tum.txt", "vilens-slam-tum.txt", "hba-tum.txt", "colmap-tum.txt")
    PATH_MAP = {
        "raw_cam0": "raw/cam0",
        "raw_cam1": "raw/cam1",
        "raw_cam2": "raw/cam2",
        "raw_imu": "raw/imu.csv",
        "raw_lidar": "raw/lidar-clouds",
        "raw_ros2bag": "raw/ros2bag",
        "raw_rosbag": "raw/rosbag",
        "trajectory_dir": "processed/trajectory",
        "gt_tum": "processed/trajectory/gt-tum.txt",
        "slam_poses": "processed/vilens-slam/slam-poses.csv",
        "vilens_undist_clouds": "processed/vilens-slam/undist-clouds",
        "colmap_dir": "processed/colmap",
    }
    BAG_TOPIC_TO_RAW = {
        "/alphasense_driver_ros/cam0/debayered/image/compressed": "raw/cam0",
        "/alphasense_driver_ros/cam1/debayered/image/compressed": "raw/cam1",
        "/alphasense_driver_ros/cam2/debayered/image/compressed": "raw/cam2",
        "/alphasense_driver_ros/imu": "raw/imu.csv",
        "/hesai/pandar": "raw/lidar-clouds",
    }
    COLMAP_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
    COLMAP_JSON_FILES = ("transforms_colmap.json", "transforms_colmap_scaled.json", "evo_align_results.json")
    COLMAP_CAM_DIRS = (
        "alphasense_driver_ros_cam0_debayered_image_compressed",
        "alphasense_driver_ros_cam1_debayered_image_compressed",
        "alphasense_driver_ros_cam2_debayered_image_compressed",
    )

    def __init__(self, seq_dir: Path):
        self.seq_dir = seq_dir

    def get_filepath(self, key: str) -> Path:
        return Path(self.seq_dir) / self.PATH_MAP[key]

    def get_all_filepaths(self) -> dict[str, Path]:
        return {key: self.get_filepath(key) for key in self.PATH_MAP}

    def expected_integrity_paths(self) -> list[Path]:
        """Paths that should exist for a complete sequence integrity check."""
        return (
            [self.get_filepath(f"raw_cam{cam_id}") for cam_id in self.RAW_CAM_IDS]
            + [
                self.get_filepath("raw_imu"),
                self.get_filepath("vilens_undist_clouds"),
                self.get_filepath("slam_poses"),
            ]
            + [self.get_filepath("trajectory_dir") / name for name in self.TRAJECTORY_FILES]
            + [self.get_filepath("colmap_dir") / "database.db"]
            + [self.get_filepath("colmap_dir") / "0" / name for name in self.COLMAP_SPARSE_FILES]
            + [self.get_filepath("colmap_dir") / name for name in self.COLMAP_JSON_FILES]
        )

    def load_slam_poses(self):
        """Load slam-poses.csv. Returns evo PoseTrajectory3D."""
        from oxspires_tools.trajectory.file_interfaces.vilens_slam import VilensSlamTrajReader

        return VilensSlamTrajReader(self.get_filepath("slam_poses")).read_file()

    def load_gt_poses(self):
        """Load GT TUM trajectory. Returns evo PoseTrajectory3D."""
        from oxspires_tools.trajectory.file_interfaces.tum import TUMTrajReader

        return TUMTrajReader(str(self.get_filepath("gt_tum"))).read_file()

    def load_camera_timestamps(self, cam_id: int) -> list[TimeStamp]:
        """Load TimeStamp objects for all images in a camera."""
        timestamps = []
        cam_dir = self.get_filepath(f"raw_cam{cam_id}")
        for img_path in sorted(cam_dir.glob("*.jpg")):
            try:
                ts = TimeStamp(t_string=img_path.stem)
                timestamps.append(ts)
            except (AssertionError, ValueError):
                continue
        return timestamps

    def parse_image_timestamps(self, cam_id: int) -> list[float]:
        """Parse float timestamps from '{sec}.{nsec}.jpg' filenames."""
        return [float(ts.t_float128) for ts in self.load_camera_timestamps(cam_id)]

    def load_cloud_timestamps(self) -> list[TimeStamp]:
        """Load TimeStamp objects for all undistorted LiDAR clouds."""
        timestamps = []
        lidar_dir = self.get_filepath("vilens_undist_clouds")
        for pcd_path in sorted(lidar_dir.glob("*.pcd")):
            try:
                # Parse cloud_{sec}_{nsec} → {sec}.{nsec}
                if not pcd_path.stem.startswith("cloud_"):
                    continue
                parts = pcd_path.stem[6:].rsplit("_", 1)
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                    continue
                if len(parts[0]) != 10 or len(parts[1]) != 9:
                    continue
                timestamp_str = f"{parts[0]}.{parts[1]}"
                ts = TimeStamp(t_string=timestamp_str)
                timestamps.append(ts)
            except ValueError:
                continue
        return timestamps

    def check_image_lidar_sync(self, cam_id: int = 0, tolerance_sec: float = 0.0) -> bool:
        """Check if undistorted cloud timestamps are synchronized with camera images."""
        image_timestamps = self.load_camera_timestamps(cam_id)
        lidar_timestamps = self.load_cloud_timestamps()

        if not image_timestamps:
            logger.error(f"No images found in raw/cam{cam_id}")
            return False

        if not lidar_timestamps:
            logger.error("No LiDAR point clouds found in processed/vilens-slam/undist-clouds")
            return False

        logger.info(f"Found {len(image_timestamps)} images, {len(lidar_timestamps)} LiDAR clouds")

        unmatched: List[TimeStamp] = []
        if tolerance_sec == 0.0:
            image_ts_set = {ts.t_string for ts in image_timestamps}
            for lidar_ts in lidar_timestamps:
                if lidar_ts.t_string not in image_ts_set:
                    unmatched.append(lidar_ts)
        else:
            image_floats = [float(ts.t_float128) for ts in image_timestamps]
            for lidar_ts in lidar_timestamps:
                lidar_float = float(lidar_ts.t_float128)
                _, diff, _ = find_closest_in_sorted(image_floats, lidar_float)
                if diff > tolerance_sec:
                    unmatched.append(lidar_ts)

        matched_count = len(lidar_timestamps) - len(unmatched)
        logger.info(
            f"Matched: {matched_count / len(lidar_timestamps) * 100:.2f}% ({matched_count}/{len(lidar_timestamps)}) LiDAR timestamps"
        )

        if unmatched:
            logger.error(f"Unmatched LiDAR timestamps ({len(unmatched)}):")
            for ts in unmatched[:10]:
                logger.error(f"  {ts.t_string}")
            if len(unmatched) > 10:
                logger.error(f"  ... and {len(unmatched) - 10} more")
            return False

        logger.info("All LiDAR timestamps are synchronized with images")
        return True
