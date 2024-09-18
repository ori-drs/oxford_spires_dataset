from pathlib import Path

from lidar_cloud_eval import evaluate_lidar_cloud
from utils import convert_e57_folder_to_pcd_folder


class ReconstructionBenchmark:
    def __init__(self, project_folder):
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.image_folder = self.project_folder / "images"
        self.gt_individual_folder = self.project_folder / "gt_clouds"
        self.individual_clouds_folder = self.project_folder / "lidar_clouds"
        self.output_folder = self.project_folder / "output"
        # TODO: check lidar cloud folder has viewpoints and is pcd, check gt folder is pcd, check image folder is jpg/png

    def evaluate_lidar_clouds(self):
        evaluate_lidar_cloud(self.output_folder, self.individual_clouds_folder, self.gt_individual_folder)


if __name__ == "__main__":
    gt_cloud_folder_e57_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_e57"
    gt_cloud_folder_pcd_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_clouds"
    convert_e57_folder_to_pcd_folder(gt_cloud_folder_e57_path, gt_cloud_folder_pcd_path)
    project_folder = "/home/oxford_spires_dataset/data/2024-03-13-maths_1"
    recon_benchmark = ReconstructionBenchmark(project_folder)
    recon_benchmark.evaluate_lidar_clouds()
