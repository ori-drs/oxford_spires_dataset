from pathlib import Path

from lidar_cloud_eval import evaluate_lidar_cloud
from utils import convert_e57_folder_to_pcd_folder

from oxford_spires_utils.bash_command import print_with_colour
from oxford_spires_utils.point_cloud import merge_downsample_vilens_slam_clouds
from spires_cpp import convertOctreeToPointCloud, processPCDFolder


class ReconstructionBenchmark:
    def __init__(self, project_folder):
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.image_folder = self.project_folder / "images"
        self.gt_individual_folder = self.project_folder / "gt_clouds"
        self.individual_clouds_folder = self.project_folder / "lidar_clouds"
        self.output_folder = self.project_folder / "output"
        # TODO: check lidar cloud folder has viewpoints and is pcd, check gt folder is pcd, check image folder is jpg/png
        self.octomap_resolution = 0.1
        self.cloud_downsample_voxel_size = 0.05

    def process_gt_cloud(self):
        print_with_colour("Creating Octree and merged cloud from ground truth clouds")
        self.gt_octree_path = self.output_folder / "gt_cloud.bt"
        self.gt_cloud_merged_path = self.output_folder / "gt_cloud_merged.ply"
        processPCDFolder(str(self.gt_individual_folder), self.octomap_resolution, str(self.gt_octree_path))
        gt_cloud_free_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_free.pcd"))
        gt_cloud_occ_path = str(Path(self.gt_octree_path).with_name(f"{Path(self.gt_octree_path).stem}_occ.pcd"))
        convertOctreeToPointCloud(str(self.gt_octree_path), str(gt_cloud_free_path), str(gt_cloud_occ_path))
        self.gt_cloud_merged = merge_downsample_vilens_slam_clouds(
            self.gt_individual_folder, self.cloud_downsample_voxel_size, self.gt_cloud_merged_path
        )

    def evaluate_lidar_clouds(self):
        evaluate_lidar_cloud(
            self.output_folder, self.individual_clouds_folder, self.gt_individual_folder, self.octomap_resolution
        )


if __name__ == "__main__":
    gt_cloud_folder_e57_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_e57"
    gt_cloud_folder_pcd_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_clouds"
    convert_e57_folder_to_pcd_folder(gt_cloud_folder_e57_path, gt_cloud_folder_pcd_path)
    project_folder = "/home/oxford_spires_dataset/data/2024-03-13-maths_1"
    recon_benchmark = ReconstructionBenchmark(project_folder)
    recon_benchmark.process_gt_cloud()
    recon_benchmark.evaluate_lidar_clouds()
