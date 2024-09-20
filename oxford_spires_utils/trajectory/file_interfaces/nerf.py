import glob
import json
from copy import deepcopy
from pathlib import Path

import evo
import numpy as np
from evo.core.trajectory import PosePath3D

from .base import BasicTrajReader, BasicTrajWriter
from .timestamp import TimeStamp


class NeRFTrajUtils:
    @staticmethod
    def get_t_float128_from_fname(fname):
        file_name = Path(fname).name
        t_string = file_name.rsplit(".", 1)[0]
        t_float128 = TimeStamp(t_string=t_string).t_float128
        return t_float128


class NeRFTrajReader(BasicTrajReader):
    """
    Read trajectory file in NeRF format
    """

    def __init__(self, file_path, nerf_reader_valid_folder_path, nerf_reader_sort_timestamp, **kwargs):
        super().__init__(file_path)
        self.valid_folder_path = nerf_reader_valid_folder_path
        self.sort_timestamp = nerf_reader_sort_timestamp

    def read_file(self):
        """
        Read NeRF trajectory file (transforms.json)
        @return: PosePath3D from evo
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            out = json.load(f)

        timestamps = []
        poses_se3 = []
        for frame in out["frames"]:
            if self.valid_folder_path != "":
                if not frame["file_path"].startswith(self.valid_folder_path):
                    continue
            t_float128 = NeRFTrajUtils.get_t_float128_from_fname(frame["file_path"])
            T = np.array(frame["transform_matrix"])
            assert T.shape == (4, 4)
            assert np.allclose(T[3, :], np.array([0, 0, 0, 1]))
            timestamps.append(t_float128)
            poses_se3.append(T)

        timestamps = np.array(timestamps)
        poses_se3 = np.array(poses_se3)
        if self.sort_timestamp:
            sort_idx = np.argsort(timestamps)
            timestamps = timestamps[sort_idx]
            poses_se3 = poses_se3[sort_idx]

        return evo.core.trajectory.PoseTrajectory3D(poses_se3=poses_se3, timestamps=timestamps)


class NeRFTrajWriter(BasicTrajWriter):
    """
    write trajectory file in NeRF format (transforms.json)
    """

    def __init__(
        self,
        file_path,
        nerf_writer_template_output_path,
        nerf_writer_template_empty,
        nerf_writer_new_image_dir,
        nerf_writer_new_depth_dir,
        nerf_writer_new_img_ext,
        nerf_writer_new_depth_ext,
        nerf_writer_new_file_prefix,
        nerf_writer_new_file_suffix,
        nerf_writer_img_folder,
        nerf_writer_is_fisheye,
        nerf_writer_intrinsics,
        nerf_writer_distortion_coeffs,
        nerf_writer_img_resolution,
        **kwargs,
    ):
        super().__init__(file_path)
        self.template_output_path = nerf_writer_template_output_path
        self.template_empty = nerf_writer_template_empty
        self.new_image_dir = nerf_writer_new_image_dir
        self.new_depth_dir = nerf_writer_new_depth_dir
        self.new_img_ext = nerf_writer_new_img_ext
        self.new_depth_ext = nerf_writer_new_depth_ext
        self.new_file_prefix = nerf_writer_new_file_prefix
        self.new_file_suffix = nerf_writer_new_file_suffix
        self.img_folder = nerf_writer_img_folder
        self.is_fisheye = nerf_writer_is_fisheye
        self.intrinsics = nerf_writer_intrinsics
        self.distortion_coeffs = nerf_writer_distortion_coeffs
        self.img_resolution = nerf_writer_img_resolution

    def write_file(self, pose):
        if self.template_empty:
            # create a new template file
            meta = self.get_new_dict(pose)
        else:
            # only replace the transform matrix in the template file
            template_meta = self.read_template_file()
            meta = self.replace_transform_matrix(pose, template_meta)
            # self.add_file_path_from_timestamp(pose, template_meta, image_dir=image_dir, image_ext=image_ext, prefix=prefix, suffix=suffix)
        if self.img_folder != "":
            meta = self.filter_pose_without_image(meta, self.img_folder)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    def get_new_dict(self, pose: PosePath3D):
        """
        Create a new dict for NeRF trajectory file (transforms.json)
        @params pose: (new) trajectory
        @return: dict
        """
        if self.is_fisheye:
            camera_dict = {"camera_model": "OPENCV_FISHEYE"}
        else:
            camera_dict = {}
        camera_intrinsics_dict = {
            "fl_x": self.intrinsics[0],
            "fl_y": self.intrinsics[1],
            "cx": self.intrinsics[2],
            "cy": self.intrinsics[3],
            "w": self.img_resolution[0],
            "h": self.img_resolution[1],
            "k1": self.distortion_coeffs[0],
            "k2": self.distortion_coeffs[1],
            "k3": self.distortion_coeffs[2],
            "k4": self.distortion_coeffs[3],
        }
        print("WARNING: camera intrinsics is hard coded: ", camera_intrinsics_dict)
        frames = []
        for i in range(pose.num_poses):
            t_string = TimeStamp(t_float128=pose.timestamps[i]).t_string
            f_name_img = self.new_file_prefix + t_string + self.new_file_suffix + "." + self.new_img_ext
            frame = {
                "file_path": f"{Path(self.new_image_dir) / f_name_img}",
                "transform_matrix": pose.poses_se3[i].tolist(),
            }
            frame = {**camera_intrinsics_dict, **frame}
            if self.new_depth_dir is not None:
                f_name_depth = self.new_file_prefix + t_string + self.new_file_suffix + "." + self.new_depth_ext
                frame["depth_file_path"] = f"{Path(self.new_depth_dir) / f_name_depth}"
            frames.append(frame)
        camera_dict["frames"] = frames
        return camera_dict

    def replace_transform_matrix(self, pose: PosePath3D, template_meta: dict):
        """
        Replace the transform matrix in template json with the poses in pose
        @params pose: (new) trajectory
        @params template_meta: template json file
        """
        meta = deepcopy(template_meta)
        assert (
            len(meta["frames"]) == pose.num_poses
        ), "Number of frames in template file does not match the number of poses"
        for i in range(pose.num_poses):
            # assume order of frames in template file is the same as the order of poses
            assert pose.timestamps[i] == NeRFTrajUtils.get_t_float128_from_fname(
                meta["frames"][i]["file_path"]
            ), "Timestamps of frames in template file does not match the timestamps of poses"
            meta["frames"][i]["transform_matrix"] = pose.poses_se3[i].tolist()
        return meta

    def read_template_file(self):
        with open(self.template_output_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.check_duplicated_timestamp(meta)
        return meta

    def check_duplicated_timestamp(self, meta):
        fnames = []
        for frames in meta["frames"]:
            fname = Path(frames["file_path"]).name
            fnames.append(fname)
        assert len(fnames) == len(set(fnames)), "Does not support duplicated file names in template file"

    def filter_pose_without_image(self, meta, img_folder):
        image_files = glob.glob(img_folder + "/*." + self.new_img_ext)
        image_files = [Path(f).name for f in image_files]
        new_meta = deepcopy(meta)
        for frame in meta["frames"]:
            if Path(frame["file_path"]).name not in image_files:
                new_meta["frames"].remove(frame)
        print(
            f"Filter {len(meta['frames']) - len(new_meta['frames'])} poses without image. {len(new_meta['frames'])} poses left."
        )
        return new_meta
