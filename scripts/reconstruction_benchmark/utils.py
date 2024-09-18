from pathlib import Path

from tqdm import tqdm

from oxford_spires_utils.point_cloud import convert_e57_to_pcd


def convert_e57_folder_to_pcd_folder(e57_folder, pcd_folder):
    Path(pcd_folder).mkdir(parents=True, exist_ok=True)
    e57_files = list(Path(e57_folder).glob("*.e57"))
    pbar = tqdm(e57_files)
    print(f"Converting {len(e57_files)} E57 files in {e57_folder} to PCD files in {pcd_folder}")
    for e57_file in pbar:
        pcd_file = Path(pcd_folder) / (e57_file.stem + ".pcd")
        pbar.set_description(f"Processing {e57_file.name}")
        convert_e57_to_pcd(e57_file, pcd_file, pcd_lib="pypcd4")
