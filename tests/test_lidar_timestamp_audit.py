import numpy as np
import pytest

from oxspires_tools.lidar_undistortion.io import point_timestamps_ns, scan_time_bounds_ns


def test_point_timestamps_ns_absolute_seconds():
    cloud = np.array([(0.5,), (0.50000025,)], dtype=[("timestamp", "f8")])

    timestamps_ns = point_timestamps_ns(cloud)

    np.testing.assert_array_equal(timestamps_ns, [500_000_000, 500_000_250])


def test_point_timestamps_ns_relative_to_scan_start():
    cloud = np.array([(0,), (125_440,)], dtype=[("t", "u4")])

    timestamps_ns = point_timestamps_ns(cloud, scan_start_ns=1_710_000_000_000_000_000)

    np.testing.assert_array_equal(
        timestamps_ns,
        [1_710_000_000_000_000_000, 1_710_000_000_000_125_440],
    )


def test_scan_time_bounds_uses_all_points():
    cloud = np.array([(300,), (100,), (250,)], dtype=[("t", "u4")])

    start_ns, end_ns = scan_time_bounds_ns(cloud, scan_start_ns=1_000)

    assert start_ns == 1_100
    assert end_ns == 1_300


def test_relative_timestamps_require_scan_start():
    cloud = np.array([(0,)], dtype=[("t", "u4")])

    with pytest.raises(ValueError, match="scan_start_ns"):
        point_timestamps_ns(cloud)


def test_timestamp_bounds_reject_empty_cloud():
    cloud = np.array([], dtype=[("timestamp", "f8")])

    with pytest.raises(ValueError, match="empty point cloud"):
        scan_time_bounds_ns(cloud)
