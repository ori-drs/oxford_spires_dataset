from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from .pose_convention import PoseConvention


@dataclass
class TrajConfigOptional:
    """
    Optional parameters for TrajConfig
    """

    tum_reader_type: Literal["custom", "evo"] = "custom"
    tum_custom_reader_prefix: str = ""  # prefix for the timestamp in tum pose file
    tum_custom_reader_suffix: str = ""  # suffix for the timestamp in tum pose file
    nerf_reader_valid_folder_path: str = ""  # path to the valid folder
    nerf_reader_sort_timestamp: bool = False  # whether to sort the timestamp in the valid folder
    nerf_writer_template_output_path: str = ""  # path to the template json file; only transform_matrix will be replaced
    nerf_writer_template_empty: bool = False  # if true, no template json file is used
    nerf_writer_new_image_dir: str = ""  #  rgb image folder when no template json is used
    nerf_writer_new_depth_dir: str = ""  # depth image folder when no template json is used
    nerf_writer_new_img_ext: str = "jpg"  # image file extension when no template json is used
    nerf_writer_new_depth_ext: str = "png"  # depth file extension when no template json is used
    nerf_writer_new_file_prefix: str = ""  # prefix for the timestamp in tum pose file when no template json is used
    nerf_writer_new_file_suffix: str = ""  # suffix for the timestamp in tum pose file when no template json is used
    nerf_writer_img_folder: str = ""  # when provided, filter poses with no corresponding image in the folder
    nerf_writer_is_fisheye: bool = False  # whether the camera is fisheye
    nerf_writer_intrinsics: List[float] = field(default_factory=list)  # [fx, fy, cx, cy]
    nerf_writer_distortion_coeffs: List[float] = field(default_factory=list)  # [k1, k2, k3, k4]
    nerf_writer_img_resolution: List[int] = field(default_factory=list)  # [width, height]

    def __post_init__(self):
        # ensure that there are no extra parameters
        assert len(self.__dict__) == 18


@dataclass
class TrajConfig:
    """
    Configuration for trajectory conversion
    """

    accepted_file_formats = ["tum", "nerf", "colmap", "vilens_slam"]
    accepted_pose_conventions = ["robotics", "colmap", "nerf", "vision", "graphics", "blender"]
    accepted_file_extension = {"tum": "txt", "nerf": "json", "colmap": "txt", "vilens_slam": "csv"}

    file_format: Literal[tuple(accepted_file_formats)]
    file_path: str  # path to the trajectory file
    pose_convention: Literal[tuple(accepted_pose_conventions)]  # pose convention of the trajectory file
    visualise: bool = False  # whether to visualise the trajectory using open3d

    optional: Optional[TrajConfigOptional] = TrajConfigOptional()

    def __post_init__(self):
        assert self.file_format in self.accepted_file_formats
        assert self.pose_convention in self.accepted_pose_conventions
        self.file_path = Path(self.file_path).absolute()
        self.check_file_path(self.file_path)
        # check if there are any extra parameters
        assert len(self.__dict__) == 5

    def check_file_path(self, file_path):
        """
        The file name must have the format <name>_<pose convention>.<extension>
        # e.g. hbac_slam_poses_robotics.txt has pose convention of "robotics"
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert isinstance(file_path, Path)
        # assert file_path.exists(), f"File {file_path} does not exist"
        file_name = file_path.name
        assert file_name.endswith(self.accepted_file_extension[self.file_format]), (
            f"File {file_path} does not have extension {self.accepted_file_extension[self.file_format]}"
        )
        assert file_name.find("_") != -1, f"File name {file_name} should have the format <name>_<pose convention>"
        file_pose_convention = file_name.split("_")[-1].split(".")[0]

        if file_pose_convention not in self.accepted_pose_conventions:
            print(f"file_pose_convention '{file_pose_convention}' does not have a valid pose convention")
            input("If you are sure, press Enter to continue...")
        if file_pose_convention != self.pose_convention:
            print(f"WARNING: File name {file_name} has different pose convention as in config")

    def rename_file_path_pose_convention(self, pose_convention: PoseConvention):
        """
        Rename self.file_path with a new pose convention
        e.g. hbac_slam_poses_robotics.txt -> hbac_slam_poses_colmap.txt
        This function does not transform the pose convention of the trajectory data. It only renames the file.
        """
        self.check_file_path(self.file_path)
        assert pose_convention in self.accepted_pose_conventions, f"Pose convention {pose_convention} is not supported"
        file_name = Path(self.file_path).name
        # replace the rightmost occurrence of the pose convention in the file name
        new_file_name = (
            file_name.rsplit("_", 1)[0] + f"_{pose_convention}.{self.accepted_file_extension[self.file_format]}"
        )
        new_file_path = Path(self.file_path).parent / new_file_name
        # new_file_path = file_name.replace(self.pose_convention, pose_convention, 1)
        self.check_file_path(new_file_path)
        self.file_path = new_file_path


@dataclass
class ProcessingConfig:
    """
    Configuration for processing the trajectory
    """

    # start timestamp of the trajectory
    start_timestamp: float = 0.0

    # end timestamp of the trajectory
    end_timestamp: float = float("inf")

    # variance of rotation for odometry noise
    odom_noise_var_R: float = 0.0

    # variance of translation for odometry noise
    odom_noise_var_t: float = 0.0

    # extra transform to apply to the trajectory
    extra_transform_t_xyz_q_xyzw: List[float] = field(default_factory=list)

    # when not zero, the trajectory will be split into submaps with the given radius
    submap_radius: float = 0.0

    def __post_init__(self):
        assert self.start_timestamp < self.end_timestamp
        # check if there are any extra parameters
        assert len(self.__dict__) == 6


@dataclass
class TrajHandlerConfig:
    input_traj: TrajConfig
    output_traj: TrajConfig
    # processing: ProcessingConfig
    processing: Optional[ProcessingConfig] = ProcessingConfig()
