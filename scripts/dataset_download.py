import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

dataset_urls = {"2024-03-13-roq-1": "https://huggingface.co/datasets/yifutao/OxSpires/resolve/main/2024-03-13-roq-1/"}

file_lists = ["images.zip", "lidar_slam.zip", "gt/individual_e57_clouds.zip", "T_gt_lidar.txt"]
for dataset in dataset_urls:
    output_folder = Path(__file__).parent.parent / "data" / dataset
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for file in file_lists:
        full_url = dataset_urls[dataset] + file
        save_path = (output_folder / file).parent
        logger.info(f"Downloading {full_url} to {save_path}")
        subprocess.run(["wget", "-P", save_path, full_url], check=False)

    files = list(output_folder.rglob("*.zip"))
    for file in files:
        logger.info(f"Unzipping {file}")
        shutil.unpack_archive(file, file.parent)
        file.unlink()
    logger.info(f'Complete downloading "{dataset}" to {output_folder}')
