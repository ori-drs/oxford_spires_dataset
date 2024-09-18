from lidar_cloud_eval import evaluate_lidar_cloud
from utils import convert_e57_folder_to_pcd_folder

if __name__ == "__main__":
    input_cloud_folder_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/individual_clouds"
    gt_cloud_folder_e57_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_e57"
    gt_cloud_folder_pcd_path = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/gt_individual_pcd"
    project_folder = "/home/oxford_spires_dataset/data/2024-03-13-maths_1/recon_benchmark"
    convert_e57_folder_to_pcd_folder(gt_cloud_folder_e57_path, gt_cloud_folder_pcd_path)
    evaluate_lidar_cloud(project_folder, input_cloud_folder_path, gt_cloud_folder_pcd_path)
