import numpy as np
import pytest

from oxford_spires_utils.trajectory.file_interfaces import TimeStamp


def test_init_with_sec_nsec():
    t = TimeStamp(sec=123, nsec=456)
    assert t.sec == 123
    assert t.nsec == 456
    assert t.t_string == "123.000000456"
    assert t.t_float128 == np.float128("123.000000456")


def test_init_with_t_string():
    t_string = "9876543219.123456789"
    t = TimeStamp(t_string=t_string)
    assert t.sec == 9876543219
    assert t.nsec == 123456789
    assert t.t_string == t_string
    assert t.t_float128 == np.float128(t_string)


def test_init_with_t_float128():
    t_string = "9876543219.123456789"
    t_float128 = np.float128(t_string)
    t = TimeStamp(t_float128=t_float128)
    assert t.sec == 9876543219
    assert t.nsec == 123456789
    assert t.t_string == t_string
    assert t.t_float128 == t_float128


def test_init_with_t_float128_decimal():
    t_float128 = np.float128("123.456")
    t = TimeStamp(t_float128=t_float128)
    assert t.sec == 123
    assert t.nsec == 456000000
    assert t.t_string == "123.456000000"


def test_exceptions():
    with pytest.raises(ValueError):
        TimeStamp(sec=123)
    with pytest.raises(ValueError):
        TimeStamp(nsec=456)
    with pytest.raises(ValueError):
        TimeStamp()
    with pytest.raises(AssertionError):
        TimeStamp(sec=12345678901, nsec=456)
    with pytest.raises(AssertionError):
        TimeStamp(t_string="123.12345678")
    with pytest.raises(AssertionError):
        TimeStamp(t_string="123.1234567890")
    with pytest.raises(AssertionError):
        TimeStamp(t_string="123.123456789a")
    with pytest.raises(AssertionError):
        TimeStamp(t_string="123.123456789.png")


def test_t_float128():
    t = TimeStamp(sec=123, nsec=456)
    assert t.t_float128 == np.float128("123.000000456")


def test_t_string():
    t = TimeStamp(sec=123, nsec=456)
    assert t.t_string == "123.000000456"


def test_get_sec_nsec_from_float128():
    sec, nsec = TimeStamp.get_sec_nsec_from_float128(np.float128("123.456"))
    assert sec == "123"
    assert nsec == "456000000"
