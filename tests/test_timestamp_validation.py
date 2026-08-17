import pytest

from oxspires_tools.trajectory.file_interfaces.timestamp import TimeStamp


def test_timestamp_rejects_negative_nanoseconds():
    with pytest.raises(AssertionError, match="should be in"):
        TimeStamp(sec=1710000000, nsec=-1)


def test_timestamp_rejects_negative_nanosecond_string():
    with pytest.raises(AssertionError):
        TimeStamp(t_string="1710000000.-00000001")


def test_timestamp_rejects_non_numeric_nanosecond_string():
    with pytest.raises(AssertionError, match="decimal digits"):
        TimeStamp(t_string="1710000000.abcdefghi")


def test_timestamp_accepts_nanosecond_range_boundaries():
    assert TimeStamp(sec=1710000000, nsec=0).t_string == "1710000000.000000000"
    assert TimeStamp(sec=1710000000, nsec=999_999_999).t_string == "1710000000.999999999"

    with pytest.raises(AssertionError, match="should be in"):
        TimeStamp(sec=1710000000, nsec=1_000_000_000)
