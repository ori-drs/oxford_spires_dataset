from pathlib import Path

from oxford_spires_utils.point_cloud import convert_e57_to_pcd


def convert_e57_folder_to_pcd_folder(e57_folder, pcd_folder):
    Path(pcd_folder).mkdir(parents=True, exist_ok=True)
    e57_files = list(Path(e57_folder).glob("*.e57"))
    for e57_file in e57_files:
        pcd_file = Path(pcd_folder) / (e57_file.stem + ".pcd")
        print(f"Converting {e57_file} to {pcd_file}")
        convert_e57_to_pcd(e57_file, pcd_file, pcd_lib="pypcd4")
