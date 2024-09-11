import multiprocessing
from pathlib import Path

from oxford_spires_utils.bash_command import run_command


def run_pcd2bt(cloud_folder: str, cloud_bt_path: str, pcd2bt_path: str, get_occ_free_from_bt_path: str):
    cloud_folder, cloud_bt_path = str(cloud_folder), str(cloud_bt_path)
    pcd2bt_path, get_occ_free_from_bt_path = str(pcd2bt_path), str(get_occ_free_from_bt_path)

    pcd2bt_command = [pcd2bt_path, cloud_folder, "-s", str(cloud_bt_path)]
    pcd2bt_command = " ".join(pcd2bt_command)
    run_command(pcd2bt_command)

    saved_occ_pcd_path = str(Path(cloud_bt_path).with_name(f"{Path(cloud_bt_path).stem}_occ.pcd"))
    saved_free_pcd_path = str(Path(cloud_bt_path).with_name(f"{Path(cloud_bt_path).stem}_free.pcd"))
    get_occ_free_from_bt_command = [
        get_occ_free_from_bt_path,
        cloud_bt_path,
        "-so",
        saved_occ_pcd_path,
        "-sf",
        saved_free_pcd_path,
    ]
    print(get_occ_free_from_bt_command)
    get_occ_free_from_bt_command = " ".join(get_occ_free_from_bt_command)
    run_command(get_occ_free_from_bt_command)


input_cloud_folder_path = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw/individual_clouds"
gt_cloud_folder_path = "/media/yifu/Samsung_T71/oxford_spires/2024-03-13-maths/gt/individual_pcd"
project_folder = "/home/yifu/workspace/Spires_2025/2024-03-13-maths_1"
Path(project_folder).mkdir(parents=True, exist_ok=True)

input_cloud_bt_path = Path(project_folder) / "input_cloud.bt"
gt_cloud_bt_path = Path(project_folder) / "gt_cloud.bt"

octomap_utils_path = Path(__file__).parent.parent / "octomap_utils"
pcd2bt_path = octomap_utils_path / "build" / "pcd2bt"
get_occ_free_from_bt_path = octomap_utils_path / "build" / "get_occ_free_from_bt"
filter_pcd_with_bt_path = octomap_utils_path / "build" / "filter_pcd_with_bt"

processes = []
for cloud_folder, cloud_bt_path in zip(
    [input_cloud_folder_path, gt_cloud_folder_path], [input_cloud_bt_path, gt_cloud_bt_path, gt_cloud_bt_path]
):
    process = multiprocessing.Process(
        target=run_pcd2bt, args=(cloud_folder, cloud_bt_path, pcd2bt_path, get_occ_free_from_bt_path)
    )
    processes.append(process)

for process in processes:
    process.start()

for process in processes:
    process.join()

input_occ_pcd_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ.pcd"))
input_occ_filtered_path = str(Path(input_cloud_bt_path).with_name(f"{Path(input_cloud_bt_path).stem}_occ_filtered.pcd"))
gt_occ_pcd_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_occ.pcd"))
gt_occ_filtered_path = str(Path(gt_cloud_bt_path).with_name(f"{Path(gt_cloud_bt_path).stem}_occ_filtered.pcd"))
filter_pcd_command = [str(filter_pcd_with_bt_path), str(gt_cloud_bt_path), input_occ_pcd_path, input_occ_filtered_path]
filter_pcd_command = " ".join(filter_pcd_command)
run_command(filter_pcd_command)
filter_pcd_command = [str(filter_pcd_with_bt_path), str(input_cloud_bt_path), gt_occ_pcd_path, gt_occ_filtered_path]
filter_pcd_command = " ".join(filter_pcd_command)
run_command(filter_pcd_command)
