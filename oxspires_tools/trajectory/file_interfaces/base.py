# from nerfstudio.utils.io import read_json, write_json
from abc import ABC


class BasicTrajReader(ABC):
    """
    Base class for trajectory reader
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        pass


class BasicTrajWriter(ABC):
    """
    Base class for trajectory writer
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def write_file(self, pose):
        pass
