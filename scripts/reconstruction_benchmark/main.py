import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from lidar_cloud_eval import evaluate_lidar_cloud
from mvs import rescale_openmvs_cloud, run_openmvs
from nerf import create_nerfstudio_dir, generate_nerfstudio_config, run_nerfstudio
from sfm import rescale_colmap_json, run_colmap

from oxford_spires_utils.bash_command import print_with_colour
from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds
from oxford_spires_utils.sensor import Sensor
from oxford_spires_utils.trajectory.align import align
from oxford_spires_utils.trajectory.file_interfaces import NeRFTrajReader, VilensSlamTrajReader
from oxford_spires_utils.trajectory.utils import pose_to_ply
from oxford_spires_utils.utils import convert_e57_folder_to_pcd_folder, transform_pcd_folder
from spires_cpp import convertOctreeToPointCloud, processPCDFolder

logger = logging.getLogger(__name__)


def setup_logging():
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename=f"logs/recon_benchmark_{time}.log",  # Log file
        level=logging.DEBUG,  # Set the logging level
        format="%(asctime)s %(levelname)s %(name)s %(lineno)s: %(message)s",  # Log format
    )
    console_handler = logging.StreamHandler()  # Create a console handler
    console_handler.setLevel(logging.INFO)  # Set the logging level
    root_logger = logging.getLogger()  # Get the root logger
    root_logger.addHandler(console_handler)  # Add the console handler to the logger


class ReconstructionBenchmark:
    def __init__(self, project_folder, sensor):
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.sensor = sensor
        self.camera_for_alignment = "cam_front"
        self.image_folder = self.project_folder / "images"
        self.gt_individual_folder = self.project_folder / "gt_clouds"
        self.individual_clouds_folder = self.project_folder / "lidar_clouds"
        self.output_folder = self.project_folder / "outputs"
        self.lidar_output_folder = self.output_folder / "lidar"
        self.lidar_output_folder.mkdir(exist_ok=True, parents=True)
        self.colmap_output_folder = self.output_folder / "colmap"
        self.colmap_output_folder.mkdir(exist_ok=True, parents=True)
        # TODO: check lidar cloud folder has viewpoints and is pcd, check gt folder is pcd, check image folder is jpg/png
        self.octomap_resolution = 0.1
        self.cloud_downsample_voxel_size = 0.05
        self.gt_octree_path = self.output_folder / "gt_cloud.bt"
        self.gt_cloud_merged_path = self.output_folder / "gt_cloud_merged.pcd"

        self.colmap_sparse_folder = self.colmap_output_folder / "sparse" / "0"
        self.openmvs_bin = "/usr/local/bin/OpenMVS"
        self.mvs_output_folder = self.output_folder / "mvs"
        self.mvs_output_folder.mkdir(exist_ok=True, parents=True)
        self.mvs_max_image_size = 600

        self.ns_data_dir = self.output_folder / "nerfstudio" / self.project_folder.name
        self.metric_json_filename = "transforms_metric.json"
        self.ns_model_dir = self.ns_data_dir / "trained_models"
        logger.info(f"Project folder: {self.project_folder}")

    def process_gt_cloud(self):
        print_with_colour("Creating Octree and merged cloud from ground truth clouds")
        processPCDFolder(str(self.gt_individual_folder), self.octomap_resolution, str(self.gt_octree_path))
        gt_cloud_free_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_free.pcd"))
        gt_cloud_occ_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_occ.pcd"))
        convertOctreeToPointCloud(str(self.gt_octree_path), str(gt_cloud_free_path), str(gt_cloud_occ_path))
        _ = merge_downsample_vilens_slam_clouds(
            self.gt_individual_folder, self.cloud_downsample_voxel_size, self.gt_cloud_merged_path
        )

    def evaluate_lidar_clouds(self):
        evaluate_lidar_cloud(
            self.lidar_output_folder,
            self.individual_clouds_folder,
            self.gt_octree_path,
            self.gt_cloud_merged_path,
            self.octomap_resolution,
        )

    def tranform_lidar_clouds(self, transform_matrix_path=None):
        if transform_matrix_path is None:
            transform_matrix_path = self.project_folder / "T_gt_lidar.txt"
        assert transform_matrix_path.exists(), f"Transform matrix not found at {transform_matrix_path}"
        transform_matrix = np.loadtxt(transform_matrix_path)
        new_individual_clouds_folder = self.project_folder / "lidar_clouds_transformed"
        transform_pcd_folder(self.individual_clouds_folder, new_individual_clouds_folder, transform_matrix)
        self.individual_clouds_folder = new_individual_clouds_folder

    def run_colmap(self):
        run_colmap(self.image_folder, self.colmap_output_folder)
        create_nerfstudio_dir(self.colmap_output_folder, self.ns_data_dir, self.image_folder)

    def run_openmvs(self):
        # check if multiple sparse folders exist
        num_sparse_folders = len(list(self.colmap_sparse_folder.glob("*")))
        if num_sparse_folders > 1:
            print_with_colour(f"Multiple sparse folders found in {self.colmap_output_folder}. Using the first one.")
        run_openmvs(
            self.image_folder,
            self.colmap_output_folder,
            self.colmap_sparse_folder,
            self.mvs_output_folder,
            self.mvs_max_image_size,
            self.openmvs_bin,
        )

    def compute_sim3(self):
        lidar_slam_traj_file = self.project_folder / "slam_poses_robotics.csv"
        colmap_traj_file = self.colmap_output_folder / "transforms.json"
        rescaled_colmap_traj_file = self.colmap_output_folder / self.metric_json_filename  # TODO refactor
        lidar_slam_traj = VilensSlamTrajReader(lidar_slam_traj_file).read_file()
        camera_alignment = self.sensor.get_camera(self.camera_for_alignment)
        valid_folder_path = "images/" + Sensor.convert_camera_topic_to_folder_name(camera_alignment.topic)
        logger.info(f'Loading only "{self.camera_for_alignment}" with directory "{valid_folder_path}" from json file')
        # colmap_traj = NeRFTrajReader(colmap_traj_file).read_file()
        colmap_traj_single_cam = NeRFTrajReader(colmap_traj_file, valid_folder_path).read_file()
        pose_to_ply(colmap_traj_single_cam, self.colmap_output_folder / "colmap_traj_single_cam.ply", [0.0, 0.0, 1.0])
        T_cam_lidar = camera_alignment.T_cam_lidar_overwrite  # TODO refactor
        T_base_lidar = self.sensor.tf.get_transform("base", "lidar")
        T_base_cam = T_base_lidar @ np.linalg.inv(T_cam_lidar)
        # T_WB @ T_BC = T_WC
        lidar_slam_traj_cam_frame = deepcopy(lidar_slam_traj)
        lidar_slam_traj_cam_frame.transform(T_base_cam, right_mul=True)
        # T_lidar_colmap = align(lidar_slam_traj, colmap_traj_single_cam, self.colmap_output_folder)
        T_lidar_colmap = align(lidar_slam_traj_cam_frame, colmap_traj_single_cam, self.colmap_output_folder)
        rescale_colmap_json(colmap_traj_file, T_lidar_colmap, rescaled_colmap_traj_file)
        mvs_cloud_file = self.mvs_output_folder / "scene_dense_nerf_world.ply"
        scaled_mvs_cloud_file = self.mvs_output_folder / "scene_dense_nerf_world_scaled.ply"
        rescale_openmvs_cloud(mvs_cloud_file, T_lidar_colmap, scaled_mvs_cloud_file)
        rescaled_colmap_traj = NeRFTrajReader(rescaled_colmap_traj_file).read_file()
        pose_to_ply(rescaled_colmap_traj, self.colmap_output_folder / "rescaled_colmap_traj.ply", [0.0, 1.0, 0.0])
        pose_to_ply(lidar_slam_traj, self.colmap_output_folder / "lidar_slam_traj.ply", [1.0, 0.0, 0.0])
        ns_metric_json_file = self.ns_data_dir / self.metric_json_filename
        if not ns_metric_json_file.exists():
            ns_metric_json_file.symlink_to(rescaled_colmap_traj_file)  # TODO remove old ones?

    def run_nerfstudio(self, method="nerfacto", json_filename="transforms_metric.json"):
        assert self.ns_data_dir.exists(), f"nerfstudio directory not found at {self.ns_data_dir}"
        ns_config = generate_nerfstudio_config(method, self.ns_data_dir / json_filename, self.ns_model_dir)
        run_nerfstudio(ns_config)


if __name__ == "__main__":
    setup_logging()
    logger.info("Starting Reconstruction Benchmark")
    with open(Path(__file__).parent.parent.parent / "config" / "sensor.yaml", "r") as f:
        sensor_config = yaml.safe_load(f)["sensor"]
        sensor = Sensor(**sensor_config)
    gt_cloud_folder_e57_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_e57"
    gt_cloud_folder_pcd_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_clouds"
    convert_e57_folder_to_pcd_folder(gt_cloud_folder_e57_path, gt_cloud_folder_pcd_path)
    project_folder = "/home/oxford_spires_dataset/data/2024-03-13-observatory-quarter-01"
    recon_benchmark = ReconstructionBenchmark(project_folder, sensor)
    recon_benchmark.process_gt_cloud()
    recon_benchmark.tranform_lidar_clouds()
    recon_benchmark.evaluate_lidar_clouds()
    recon_benchmark.run_colmap()
    recon_benchmark.run_openmvs()
    recon_benchmark.compute_sim3()
    recon_benchmark.run_nerfstudio("nerfacto", json_filename="transforms_metric.json")
    recon_benchmark.run_nerfstudio("splatfacto")
