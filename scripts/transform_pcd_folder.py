from pathlib import Path

import numpy as np

from oxspires_tools.utils import transform_pcd_folder

if __name__ == "__main__":
    pcd_folder_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/input_cloud_test"
    new_pcd_folder_path = Path(pcd_folder_path).parent / (Path(pcd_folder_path).name + "_new")

    # transform_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    se3_matrix_txt_path = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1/T_rtc_lidar"
    transform_matrix = np.loadtxt(se3_matrix_txt_path)

    transform_pcd_folder(pcd_folder_path, new_pcd_folder_path, transform_matrix)
