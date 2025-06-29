import argparse
import csv
import logging
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from mvs import rescale_openmvs_cloud, run_openmvs
from nerf import create_nerfstudio_dir, generate_nerfstudio_config, run_nerfstudio
from sfm import export_json, rescale_colmap_json, run_colmap

from oxspires_tools.bash_command import print_with_colour
from oxspires_tools.cpp import convertOctreeToPointCloud, processPCDFolder, removeUnknownPoints
from oxspires_tools.eval import get_recon_metrics_multi_thresholds, save_error_cloud
from oxspires_tools.point_cloud import merge_downsample_vilens_slam_clouds, transform_cloud_with_se3
from oxspires_tools.se3 import is_se3_matrix
from oxspires_tools.sensor import Sensor
from oxspires_tools.trajectory.align import align
from oxspires_tools.trajectory.file_interfaces import NeRFTrajReader, VilensSlamTrajReader
from oxspires_tools.trajectory.utils import pose_to_ply
from oxspires_tools.utils import convert_e57_folder_to_pcd_folder, transform_pcd_folder

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
    def __init__(self, project_folder, gt_folder, sensor):
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.gt_folder = Path(gt_folder)
        self.sensor = sensor
        self.camera_for_alignment = "cam_front"
        self.image_folder = self.project_folder / "images"
        self.individual_clouds_folder = self.project_folder / "lidar_slam" / "individual_clouds"
        self.lidar_slam_traj_file = self.project_folder / "lidar_slam" / "slam_poses.csv"
        self.output_folder = self.project_folder / "outputs"
        self.lidar_output_folder = self.output_folder / "lidar"
        self.lidar_output_folder.mkdir(exist_ok=True, parents=True)
        self.colmap_output_folder = self.output_folder / "colmap"
        self.colmap_output_folder.mkdir(exist_ok=True, parents=True)
        self.recon_benchmark_dir = self.output_folder / "recon_benchmark"
        self.recon_benchmark_dir.mkdir(exist_ok=True, parents=True)
        # TODO: check lidar cloud folder has viewpoints and is pcd, check gt folder is pcd, check image folder is jpg/png
        self.octomap_resolution = 0.1
        self.cloud_downsample_voxel_size = 0.01
        self.gt_octree_path = self.recon_benchmark_dir / "gt_cloud.bt"
        self.gt_cloud_merged_path = self.recon_benchmark_dir / "gt_cloud_merged.pcd"
        self.gt_cloud_individual_e57_folder = self.gt_folder / "individual_cloud_e57"
        self.gt_cloud_individual_pcd_folder = self.gt_folder / "individual_cloud_pcd"

        self.colmap_sparse_folder = self.colmap_output_folder / "sparse"
        self.colmap_sparse_0_folder = self.colmap_sparse_folder / "0"
        self.openmvs_bin = "/usr/local/bin/OpenMVS"
        self.mvs_output_folder = self.output_folder / "mvs"
        self.mvs_output_folder.mkdir(exist_ok=True, parents=True)
        self.scaled_mvs_cloud_gt_frame_file = self.recon_benchmark_dir / "OpenMVS_dense_cloud_gt_frame.pcd"
        self.colmap_undistort_max_image_size = -1

        self.ns_data_dir = self.output_folder / "nerfstudio" / self.project_folder.name
        self.metric_json_filename = "transforms_metric.json"
        logger.info(f"Project folder: {self.project_folder}")

        self.lidar_cloud_merged_path = self.recon_benchmark_dir / "lidar_cloud_merged.pcd"
        self.lidar_occ_benchmark_file = self.recon_benchmark_dir / "lidar_occ.pcd"

    def process_gt_cloud(self):
        logger.info("Converting ground truth clouds from e57 to pcd")
        convert_e57_folder_to_pcd_folder(self.gt_cloud_individual_e57_folder, self.gt_cloud_individual_pcd_folder)
        logger.info("Creating Octree and merged cloud from ground truth clouds")
        processPCDFolder(str(self.gt_cloud_individual_pcd_folder), self.octomap_resolution, str(self.gt_octree_path))
        gt_cloud_free_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_free.pcd"))
        gt_cloud_occ_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_occ.pcd"))
        convertOctreeToPointCloud(str(self.gt_octree_path), str(gt_cloud_free_path), str(gt_cloud_occ_path))
        logger.info("Merging and downsampling ground truth clouds")
        _ = merge_downsample_vilens_slam_clouds(
            self.gt_cloud_individual_pcd_folder, self.cloud_downsample_voxel_size, self.gt_cloud_merged_path
        )

    def load_lidar_gt_transform(self, transform_matrix_path=None):
        if transform_matrix_path is None:
            transform_matrix_path = self.project_folder / "T_gt_lidar.txt"
        logger.info(f"Loading transform matrix from {transform_matrix_path}")
        assert transform_matrix_path.exists(), f"Transform matrix not found at {transform_matrix_path}"
        self.transform_matrix = np.loadtxt(transform_matrix_path)
        assert is_se3_matrix(self.transform_matrix)[0], is_se3_matrix(self.transform_matrix)[1]

    def process_lidar_clouds(self):
        logger.info("Transforming lidar clouds to the same frame as the ground truth clouds")
        new_individual_clouds_folder = self.lidar_output_folder / "lidar_clouds_transformed"
        transform_pcd_folder(self.individual_clouds_folder, new_individual_clouds_folder, self.transform_matrix)
        self.individual_clouds_folder = new_individual_clouds_folder
        logger.info("Creating Octree from transformed lidar clouds")
        lidar_cloud_octomap_file = self.lidar_output_folder / "lidar_cloud.bt"
        processPCDFolder(str(self.individual_clouds_folder), self.octomap_resolution, str(lidar_cloud_octomap_file))
        logger.info("Converting Octree to point cloud")
        lidar_cloud_free_path = Path(lidar_cloud_octomap_file).with_name(
            f"{Path(lidar_cloud_octomap_file).stem}_free.pcd"
        )
        lidar_cloud_occ_path = Path(lidar_cloud_octomap_file).with_name(
            f"{Path(lidar_cloud_octomap_file).stem}_occ.pcd"
        )
        convertOctreeToPointCloud(str(lidar_cloud_octomap_file), str(lidar_cloud_free_path), str(lidar_cloud_occ_path))
        shutil.copy(lidar_cloud_occ_path, self.lidar_occ_benchmark_file)
        logger.info("Merging and downsampling lidar clouds")
        _ = merge_downsample_vilens_slam_clouds(
            self.individual_clouds_folder, self.cloud_downsample_voxel_size, self.lidar_cloud_merged_path
        )

    def run_colmap(self, matcher="vocab_tree_matcher"):
        camera_model = "OPENCV_FISHEYE"
        run_colmap(
            self.image_folder,
            self.colmap_output_folder,
            matcher=matcher,
            camera_model=camera_model,
            max_image_size=self.colmap_undistort_max_image_size,
            loop_detection_period=5,
        )
        export_json(
            self.colmap_sparse_0_folder,
            json_file_name="transforms.json",
            output_dir=self.colmap_output_folder,
        )
        export_json(
            input_bin_dir=self.colmap_output_folder / "dense" / "sparse",
            json_file_name="transforms.json",
            output_dir=self.colmap_output_folder / "dense",
            db_file=self.colmap_output_folder / "database.db",
        )
        create_nerfstudio_dir(self.colmap_output_folder, self.ns_data_dir, self.image_folder)
        create_nerfstudio_dir(
            self.colmap_output_folder / "dense",
            self.ns_data_dir.with_name(self.ns_data_dir.name + "_undistorted"),
            self.ns_data_dir / "dense" / self.image_folder.name,
        )

    def run_openmvs(self):
        # check if multiple sparse folders exist
        num_sparse_folders = len(list(self.colmap_sparse_folder.glob("*")))
        if num_sparse_folders > 1:
            print_with_colour(f"Multiple sparse folders found in {self.colmap_output_folder}. Using the first one.")
        assert self.transform_matrix is not None, "Ground truth to lidar transform not loaded"
        mvs_cloud_file = run_openmvs(
            self.image_folder,
            self.colmap_output_folder,
            self.colmap_sparse_0_folder,
            self.mvs_output_folder,
            self.openmvs_bin,
        )
        scaled_mvs_cloud_file = self.mvs_output_folder / "OpenMVS_dense_cloud_metric.pcd"
        rescale_openmvs_cloud(mvs_cloud_file, T_lidar_colmap, scaled_mvs_cloud_file)
        transform_cloud_with_se3(scaled_mvs_cloud_file, self.transform_matrix, self.scaled_mvs_cloud_gt_frame_file)

    def compute_colmap_sim3(self):
        colmap_traj_file = self.colmap_output_folder / "transforms.json"
        rescaled_colmap_traj_file = self.colmap_output_folder / self.metric_json_filename  # TODO refactor
        lidar_slam_traj = VilensSlamTrajReader(self.lidar_slam_traj_file).read_file()
        pose_to_ply(lidar_slam_traj, self.colmap_output_folder / "lidar_slam_traj.ply", [1.0, 0.0, 0.0])
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
        self.T_lidar_colmap = align(lidar_slam_traj_cam_frame, colmap_traj_single_cam, self.colmap_output_folder)
        rescale_colmap_json(colmap_traj_file, self.T_lidar_colmap, rescaled_colmap_traj_file)
        rescaled_colmap_traj = NeRFTrajReader(rescaled_colmap_traj_file).read_file()
        pose_to_ply(rescaled_colmap_traj, self.colmap_output_folder / "rescaled_colmap_traj.ply", [0.0, 1.0, 0.0])

        ns_metric_json_file = self.ns_data_dir / self.metric_json_filename
        ns_metric_json_file.unlink(missing_ok=True)
        ns_metric_json_file.symlink_to(rescaled_colmap_traj_file)

    def run_nerfstudio(
        self,
        method="nerfacto",
        ns_data_dir=None,
        json_filename="transforms_metric.json",
        eval_mode="fraction",
        export_cloud=True,
    ):
        ns_data_dir = self.ns_data_dir if ns_data_dir is None else Path(ns_data_dir)
        ns_model_dir = ns_data_dir / "trained_models"
        assert ns_data_dir.exists(), f"nerfstudio directory not found at {ns_data_dir}"
        ns_config, ns_data_config = generate_nerfstudio_config(
            method, ns_data_dir / json_filename, ns_model_dir, eval_mode=eval_mode
        )
        metric_cloud_file = run_nerfstudio(ns_config, ns_data_config, export_cloud)
        if metric_cloud_file is not None:
            metric_cloud_gt_frame = self.recon_benchmark_dir / (metric_cloud_file.stem + "_gt_frame.pcd")
            transform_cloud_with_se3(metric_cloud_file, self.transform_matrix, metric_cloud_gt_frame)
            self.evaluate_reconstruction(metric_cloud_gt_frame)

    def evaluate_reconstruction(self, input_cloud_path, results_dir=None):
        assert Path(input_cloud_path).exists(), f"Input cloud not found at {input_cloud_path}"
        assert Path(input_cloud_path).suffix == ".pcd", "Input cloud must be a pcd file"
        assert Path(self.gt_octree_path).exists(), f"Ground truth octree not found at {self.gt_octree_path}"
        recon_thresholds = [0.03, 0.05, 0.1, 0.2]
        results_dir = self.recon_benchmark_dir if results_dir is None else Path(results_dir)
        input_cloud = o3d.io.read_point_cloud(str(input_cloud_path))
        logger.info(f"Cropping input cloud using bounding box from {self.gt_cloud_merged_path}")
        gt_bbox = o3d.io.read_point_cloud(str(self.gt_cloud_merged_path)).get_axis_aligned_bounding_box()
        input_cloud = input_cloud.crop(gt_bbox)
        logger.info(f"Downsampling filtered cloud to {self.cloud_downsample_voxel_size} m")
        input_cloud_downsampled = input_cloud.voxel_down_sample(voxel_size=self.cloud_downsample_voxel_size)
        input_cloud_downsampled_path = results_dir / (input_cloud_path.stem + "_downsampled.pcd")
        o3d.io.write_point_cloud(str(input_cloud_downsampled_path), input_cloud_downsampled)

        logger.info(f"Removing unknown points using {self.gt_octree_path}")
        filtered_input_cloud_path = results_dir / f"{Path(input_cloud_downsampled_path).stem}_filtered.pcd"
        removeUnknownPoints(str(input_cloud_downsampled_path), str(self.gt_octree_path), str(filtered_input_cloud_path))
        input_cloud_np = np.asarray(o3d.io.read_point_cloud(str(filtered_input_cloud_path)).points)
        gt_cloud_np = np.asarray(o3d.io.read_point_cloud(str(self.gt_cloud_merged_path)).points)
        recon_metrics = get_recon_metrics_multi_thresholds(input_cloud_np, gt_cloud_np, thresholds=recon_thresholds)
        error_csv_file = results_dir / f"{Path(input_cloud_path).stem}_metrics.csv"
        with open(error_csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=recon_metrics[1].keys())
            writer.writeheader()
            for metric in recon_metrics[1:]:
                writer.writerow(metric)
            writer = csv.DictWriter(f, fieldnames=recon_metrics[0].keys())
            writer.writeheader()
            writer.writerow(recon_metrics[0])

        error_cloud_file = results_dir / f"{Path(input_cloud_path).stem}_error.pcd"
        save_error_cloud(input_cloud_np, gt_cloud_np, str(error_cloud_file))


def get_args():
    parser = argparse.ArgumentParser(description="Reconstruction Benchmark")
    default_recon_config_file = Path(__file__).parent.parent.parent / "config" / "recon_benchmark.yaml"
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(default_recon_config_file),
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    recon_config_file = get_args().config_file
    logger.info("Starting Reconstruction Benchmark")
    with open(recon_config_file, "r") as f:
        recon_config = yaml.safe_load(f)["reconstruction_benchmark"]
    with open(Path(__file__).parent.parent.parent / "config" / "sensor.yaml", "r") as f:
        sensor_config = yaml.safe_load(f)["sensor"]
        sensor = Sensor(**sensor_config)

    recon_benchmark = ReconstructionBenchmark(recon_config["project_folder"], recon_config["gt_folder"], sensor)
    recon_benchmark.load_lidar_gt_transform()
    if recon_config["run_gt_cloud_processing"]:
        recon_benchmark.process_gt_cloud()
    if recon_config["run_lidar_cloud_processing"]:
        recon_benchmark.process_lidar_clouds()
    if recon_config["run_lidar_cloud_evaluation"]:
        recon_benchmark.evaluate_reconstruction(recon_benchmark.lidar_cloud_merged_path)
    if recon_config["run_colmap"]:
        recon_benchmark.run_colmap("sequential_matcher")
    if recon_config["run_colmap_sim3"]:
        recon_benchmark.compute_colmap_sim3()
    if recon_config["run_mvs"]:
        recon_benchmark.run_openmvs()
    if recon_config["run_mvs_evaluation"]:
        recon_benchmark.evaluate_reconstruction(recon_benchmark.scaled_mvs_cloud_gt_frame_file)
    if recon_config["run_nerfstudio"]:
        if recon_config["run_novel_view_synthesis_only"]:
            # fmt: off
            undistorted_ns_dir = recon_benchmark.ns_data_dir.with_name(recon_benchmark.ns_data_dir.name + "_undistorted") # noqa: F401
            recon_benchmark.run_nerfstudio("nerfacto", json_filename="transforms_train.json", eval_mode="fraction", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("nerfacto", json_filename="transforms_train_eval.json", eval_mode="filename", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("nerfacto-big", json_filename="transforms_train.json", eval_mode="fraction", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("nerfacto-big", json_filename="transforms_train_eval.json", eval_mode="filename", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("splatfacto", json_filename="transforms_train.json", eval_mode="fraction", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("splatfacto", json_filename="transforms_train_eval.json", eval_mode="filename", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("splatfacto-big", json_filename="transforms_train.json", eval_mode="fraction", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            recon_benchmark.run_nerfstudio("splatfacto-big", json_filename="transforms_train_eval.json", eval_mode="filename", ns_data_dir=undistorted_ns_dir, export_cloud=False)# noqa
            # fmt: on
        else:
            recon_benchmark.run_nerfstudio("nerfacto", json_filename="transforms_metric.json")
            recon_benchmark.run_nerfstudio("splatfacto")
