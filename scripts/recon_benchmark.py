import multiprocessing
from pathlib import Path
from typing import List

from oxford_spires_utils.bash_command import run_command


def run_pcd2bt(cloud_folder: str, cloud_bt_path: str, transform: List[float]):
    pcd2bt_command = [str(pcd2bt_path), cloud_folder, "-s", str(cloud_bt_path), "-tf", *map(str, transform)]
    pcd2bt_command = " ".join(pcd2bt_command)
    run_command(pcd2bt_command)


input_cloud_folder_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw/individual_clouds"
gt_cloud_folder_path = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual_pcd"
project_folder = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1"
Path(project_folder).mkdir(parents=True, exist_ok=True)
# tf: tx ty tz qw, qx, qy, qz
input_cloud_tf = [-85.867073, 21.640722, 0.720384, 0.766620, -0.0019915, 0.0196701, -0.641796]
gt_cloud_tf = [0, 0, 0, 1, 0, 0, 0]

input_cloud_bt_path = Path(project_folder) / "input_cloud.bt"
gt_cloud_bt_path = Path(project_folder) / "gt_cloud.bt"

octomap_utils_path = Path(__file__).parent.parent / "octomap_utils"
pcd2bt_path = octomap_utils_path / "build" / "pcd2bt"

processes = []
for cloud_folder, cloud_bt_path, transform in zip(
    [input_cloud_folder_path, gt_cloud_folder_path],
    [input_cloud_bt_path, gt_cloud_bt_path, gt_cloud_bt_path],
    [input_cloud_tf, gt_cloud_tf],
):
    process = multiprocessing.Process(target=run_pcd2bt, args=(cloud_folder, cloud_bt_path, transform))
    processes.append(process)

for process in processes:
    process.start()

for process in processes:
    process.join()
