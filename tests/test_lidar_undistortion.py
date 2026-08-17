import numpy as np

from oxspires_tools.lidar_undistortion.core import PoseBuffer


def _pose_at_x(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_pose_buffer_preserves_submicrosecond_timing_at_epoch():
    epoch_ns = 1_710_000_000_000_000_000
    timestamps_ns = [epoch_ns, epoch_ns + 300]
    poses = [_pose_at_x(0.0), _pose_at_x(3.0)]

    pose_buffer = PoseBuffer(timestamps_ns, poses)
    interpolated = pose_buffer.interpolate_batch(np.array([epoch_ns + 150], dtype=np.int64))[0]

    np.testing.assert_allclose(interpolated[:3, 3], [1.5, 0.0, 0.0], atol=1e-12)


def test_pose_buffer_clamps_queries_outside_time_range():
    epoch_ns = 1_710_000_000_000_000_000
    timestamps_ns = [epoch_ns, epoch_ns + 1_000]
    poses = [_pose_at_x(1.0), _pose_at_x(2.0)]

    pose_buffer = PoseBuffer(timestamps_ns, poses)
    interpolated = pose_buffer.interpolate_batch(
        np.array([epoch_ns - 1_000, epoch_ns + 2_000], dtype=np.int64)
    )

    np.testing.assert_allclose(interpolated[0, :3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(interpolated[1, :3, 3], [2.0, 0.0, 0.0], atol=1e-12)
