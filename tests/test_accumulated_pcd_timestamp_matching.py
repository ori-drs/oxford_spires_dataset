import numpy as np

from oxspires_tools.utils import _get_transform_at_timestamp, _timestamp_string_to_ns


def test_timestamp_string_to_ns_preserves_sub_float64_difference():
    t0 = _timestamp_string_to_ns("1710000000.000000000")
    t1 = _timestamp_string_to_ns("1710000000.000000100")

    assert t1 - t0 == 100


def test_transform_lookup_distinguishes_100ns_pose_keys():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 1.0
    transforms = {
        "1710000000.000000000": first,
        "1710000000.000000100": second,
    }

    result = _get_transform_at_timestamp(
        "1710000000.000000100", transforms, max_time_diff_camera_and_pose=0.0
    )

    np.testing.assert_array_equal(result, second)


def test_zero_tolerance_mismatch_returns_none_without_prompt(monkeypatch):
    transforms = {"1710000000.000000100": np.eye(4)}

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("timestamp lookup must not prompt for input")

    monkeypatch.setattr("builtins.input", fail_if_prompted)

    assert (
        _get_transform_at_timestamp(
            "1710000000.000000000", transforms, max_time_diff_camera_and_pose=0.0
        )
        is None
    )


def test_transform_lookup_respects_nanosecond_tolerance():
    transform = np.eye(4)
    transforms = {"1710000000.000000100": transform}

    assert (
        _get_transform_at_timestamp(
            "1710000000.000000000", transforms, max_time_diff_camera_and_pose=1e-7
        )
        is transform
    )
