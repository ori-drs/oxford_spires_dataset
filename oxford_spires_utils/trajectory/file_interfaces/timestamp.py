from decimal import Decimal

import numpy as np

SECONDS_DIGITS_MAX = 10
NSECONDS_DIGITS = 9


class TimeStamp:
    """
    Similar structure to ROS TimeStamp
    """

    def __init__(self, sec=None, nsec=None, t_float128=None, t_string=None):
        """
        # 1
        @param sec: str or int
        @param nsec: str of 9 digits or int
        # 2
        @param t_float128: timestamp as np.float128 to preserve precision
        # 3
        @param t_string: timestamp as str with 9 digits of nsec (e.g. "123.456000000")
        """
        if sec is not None:
            if nsec is None:
                raise ValueError("nsec not set")
            self.sec = sec
            self.nsec = nsec
        elif t_string is not None:
            self.sec, self.nsec = TimeStamp.get_sec_nsec_from_string(t_string)

        elif t_float128 is not None:
            self.sec, self.nsec = TimeStamp.get_sec_nsec_from_float128(t_float128)
        else:
            raise ValueError("no input")

    @property
    def sec(self):
        """
        seconds: int (< 10^10)
        """
        return self._sec

    @sec.setter
    def sec(self, value):
        if isinstance(value, str):
            value = int(value)
        assert isinstance(value, int)
        assert value < pow(10, SECONDS_DIGITS_MAX), f"sec should be less than {pow(10, SECONDS_DIGITS_MAX)}"
        self._sec = value

    @property
    def nsec(self):
        """
        nanoseconds: int (< 10^9)
        """
        return self._nsec

    @nsec.setter
    def nsec(self, value):
        if isinstance(value, str):
            value = int(value)
        assert isinstance(value, int)
        assert value < pow(10, NSECONDS_DIGITS), f"nsec {value} should be less than 10^{NSECONDS_DIGITS}"

        self._nsec = value

    @property
    def t_float128(self):
        """
        only string input can ensure precision
        @return: timestamp of type np.float128
        """

        t_float128 = np.float128(self.t_string)
        # check
        assert TimeStamp.get_string_from_t_float128(t_float128) == self.t_string, (
            f"Precision Lost: t_float128 {t_float128} should be {self.t_string}"
        )

        return t_float128

    @property
    def t_string(self):
        """
        @return: str of timestamp with 9 digits of nsec (e.g. "123.000000456")
        """
        sec_string = str(self.sec)
        nsec_string = str(self.nsec).zfill(NSECONDS_DIGITS)
        return f"{sec_string}.{nsec_string}"

    @staticmethod
    def get_sec_nsec_from_string(t_string):
        """
        @param t_string: str of timestamp
        @return: sec (string), nsec (string, 9 digits)
        """
        t = t_string.split(".")
        assert len(t) == 2, f"t_string {t_string} should have 2 parts separated by '.'"
        sec = t[0]
        nsec = t[1]
        TimeStamp.check_nsec_str(nsec)

        return sec, nsec

    @staticmethod
    def get_sec_nsec_from_float128(t_float128):
        """
        @param f: np.float128
        @return: (sec, nsec), both str, nsec is 9 digits
        """
        assert isinstance(t_float128, np.float128), (
            f"t_float128 should be of type np.float128, but is {type(t_float128)}"
        )
        sec, nsec = str(t_float128).split(".")
        nsec = nsec.ljust(NSECONDS_DIGITS, "0")
        assert len(nsec) == NSECONDS_DIGITS, f"nsec {nsec} should be {NSECONDS_DIGITS} digits"
        # print(f"debug: sec={sec}, nsec={nsec} from t_float128={str(t_float128)}")
        # TimeStampHandler.check_nsec(nsec)
        # assert TimeStampHandler.get_float128_from_sec_nsec(sec, nsec) == t_float128

        return sec, nsec

    @staticmethod
    def check_nsec_str(nsec):
        """
        @param nsec: str of 9 digits
        @return: bool
        """
        assert isinstance(nsec, str)
        assert len(nsec) == NSECONDS_DIGITS, f"nsec {nsec} should be {NSECONDS_DIGITS} digits"

    @staticmethod
    def get_string_from_t_float128(t_float_128: np.float128):
        """
        convert time from float128 to string (sec.nsec with 9 digits nsec)
        @param t_float_128: timestamp of type np.float128
        @param t_string: timestamp of format "sec.nsec", 9 digits nsec
        """
        assert isinstance(t_float_128, np.float128)
        t_decimal = Decimal(str(t_float_128))
        return f"{t_decimal:.{NSECONDS_DIGITS}f}"
