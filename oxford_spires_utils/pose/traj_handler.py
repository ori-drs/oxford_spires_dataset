from copy import deepcopy
from pathlib import Path

import evo
import numpy as np

from .file_interfaces.nerf import NeRFTrajReader, NeRFTrajWriter
from .file_interfaces.tum import TUMTrajReader, TUMTrajWriter
from .file_interfaces.vilens_slam import VilensSlamTrajReader
from .pose_convention import PoseConvention
from .traj_config import TrajHandlerConfig

# from .odom_noise import add_odom_noise


class TrajHandler:
    def __init__(self, config: TrajHandlerConfig):
        self.config = config
        self.traj_reader = self.get_traj_reader(
            config.input_traj.file_format, config.input_traj.file_path, **config.input_traj.optional.__dict__
        )
        self.traj_writer = self.get_traj_writer(
            config.output_traj.file_format, config.output_traj.file_path, **config.output_traj.optional.__dict__
        )

    def run(self):
        """
        Read the trajectory, convert the frame convention and write the trajectory
        """
        self.traj_input = self.read_traj()

        if self.config.input_traj.visualise:
            self.visualise_traj(self.traj_input)
        self.reduce_to_time_range(
            self.traj_input, self.config.processing.start_timestamp, self.config.processing.end_timestamp
        )
        self.traj_output = deepcopy(self.traj_input)

        if self.config.processing.odom_noise_var_R > 0 or self.config.processing.odom_noise_var_t > 0:
            # self.traj_output = add_odom_noise(
            #     self.traj_output, self.config.processing.odom_noise_var_R, self.config.processing.odom_noise_var_t
            # )
            raise NotImplementedError("not using odom noise to remove theseus & pytorch dependency")
        if len(self.config.processing.extra_transform_t_xyz_q_xyzw) > 0:
            self.traj_output = self.apply_transform(
                self.traj_output,
                self.config.processing.extra_transform_t_xyz_q_xyzw,
            )
        self.traj_output = self.convert_frame_convention(self.traj_output)
        if self.config.output_traj.visualise:
            self.visualise_traj(self.traj_output)
        if self.config.processing.submap_radius > 0:
            self.write_traj_submaps(self.traj_output, self.config.processing.submap_radius)
        else:
            self.write_traj(self.traj_output)

    def get_traj_reader(self, file_format: str, file_path: str, **kwargs):
        """
        Get the trajectory reader based on the file format
        """
        readers = {
            "tum": TUMTrajReader,
            "nerf": NeRFTrajReader,
            # "colmap": ColmapTrajReader,
            "vilens_slam": VilensSlamTrajReader,
        }
        assert file_format in readers.keys(), f"File format {file_format} is not supported"

        return readers[file_format](file_path, **kwargs)

    def get_traj_writer(self, file_format: str, file_path: str, **kwargs):
        """
        Get the trajectory writer based on the file format
        """
        writers = {
            "tum": TUMTrajWriter,
            "nerf": NeRFTrajWriter,
            # "colmap": ColmapTrajWriter,
            # "vilens_slam": VilensSlamTrajWriter,
        }
        assert file_format in writers.keys(), f"File format {file_format} is not supported"
        return writers[file_format](file_path, **kwargs)

    def read_traj(self):
        return self.traj_reader.read_file()

    def convert_frame_convention(self, pose):
        """
        Convert the frame convention of the pose
        Define pose = T_WB, where W is the world frame and B is the body frame
        from_convention: B frame (source frame)
        to_convention: A frame (target frame)
        @return: T_WA = T_WB @ T_BA
        """
        from_convention = self.config.input_traj.pose_convention
        to_convention = self.config.output_traj.pose_convention
        T_from_to = PoseConvention.get_transform(from_convention, to_convention)
        pose_copy = deepcopy(pose)
        pose_copy.transform(T_from_to, right_mul=True)

        return pose_copy

    def write_traj(self, pose):
        self.traj_writer.write_file(pose)

    def write_traj_submaps(self, pose, submap_radius):
        """
        partition the trajectory into submaps
        """
        submap_node_indices = [0]
        for i in range(1, pose.positions_xyz.shape[0]):
            # if current pose's distance to all the submap nodes is larger than submap_radius
            # add current pose as a new submap node
            if np.all(
                np.linalg.norm(pose.positions_xyz[i] - pose.positions_xyz[submap_node_indices], axis=1) > submap_radius
            ):
                submap_node_indices.append(i)

        print(f"Number of submaps: {len(submap_node_indices)}")
        new_traj = deepcopy(pose)
        new_traj.reduce_to_ids(submap_node_indices)
        self.visualise_traj(new_traj, axis_viz_size=20)

        pcds = []
        Path(self.config.output_traj.file_path.parent).mkdir(parents=True, exist_ok=True)
        for i in range(len(submap_node_indices)):
            # get poses indices that are within the submap radius
            submap_poses_indices = np.where(
                np.linalg.norm(pose.positions_xyz - pose.positions_xyz[submap_node_indices[i]], axis=1) < submap_radius
            )[0]
            submap_poses = deepcopy(pose)
            submap_poses.reduce_to_ids(submap_poses_indices)
            fname = (
                str(self.config.output_traj.file_path.stem) + f"_{i}" + str(self.config.output_traj.file_path.suffix)
            )

            submap_file_path = self.config.output_traj.file_path.parent / fname
            submap_traj_writer = self.get_traj_writer(
                self.config.output_traj.file_format, submap_file_path, **self.config.output_traj.optional.__dict__
            )
            print(f"Writing submap {i} to {submap_file_path} with {submap_poses.num_poses} poses")
            submap_traj_writer.write_file(submap_poses)

            colour = np.random.rand(3).tolist()

            pcd = self.getTriangleMeshfromPose(submap_poses, axis_viz_size=5, colour=colour)
            # if same coordinate exist in pcds, remove it

            pcds += pcd
        import open3d as o3d

        o3d.visualization.draw_geometries(pcds)

    def getTriangleMeshfromPose(self, pose, axis_viz_size=1.0, colour=None):
        import open3d as o3d

        assert isinstance(pose, evo.core.trajectory.PosePath3D)
        pcds = []
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        pcds.append(origin)
        # add axis to show the poses
        for i in range(pose.positions_xyz.shape[0]):
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_viz_size)
            T = pose.poses_se3[i]
            axis.transform(T)
            if colour is not None:
                assert len(colour) == 3
                axis.paint_uniform_color(colour)
            pcds.append(axis)

        # lines between each timestamp to check if timestamps are ordered
        for i in range(pose.positions_xyz.shape[0] - 1):
            line_point_1 = pose.poses_se3[i][:3, 3]
            line_point_2 = pose.poses_se3[i + 1][:3, 3]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([line_point_1, line_point_2])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            pcds.append(line)
        return pcds

    def visualise_traj(self, pose, axis_viz_size=1.0):
        import open3d as o3d

        pcds = self.getTriangleMeshfromPose(pose, axis_viz_size)
        o3d.visualization.draw_geometries(pcds)

    def reduce_to_time_range(self, pose, start_timestamp: float, end_timestamp: float):
        """
        Segment the trajectory based on the start and end timestamp
        """
        assert isinstance(pose, evo.core.trajectory.PoseTrajectory3D)
        pose.reduce_to_time_range(start_timestamp, end_timestamp)

    def apply_transform(self, pose, t_xyz_quat_xyzw, right_mul=True):
        """
        Apply a transform to the pose
        @param pose: PoseTrajectory3D
        @param t: translation xyz
        @param quat: quaternion xyzw
        @param right_mul: True if the transform is applied on the right side of the pose
        """
        assert isinstance(pose, evo.core.trajectory.PoseTrajectory3D)
        assert len(t_xyz_quat_xyzw) == 7
        t = t_xyz_quat_xyzw[:3]
        quat_xyzw = t_xyz_quat_xyzw[3:]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        transform = evo.core.transformations.quaternion_matrix(quat_wxyz)
        transform[:3, 3] = t
        pose_copy = deepcopy(pose)
        pose_copy.transform(transform, right_mul=right_mul)
        return pose_copy
