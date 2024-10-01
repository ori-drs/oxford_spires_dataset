import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

repo_id = "ori-drs/oxford_spires_dataset"
dataset_sequences = ["2024-03-13-observatory-quarter-01", "2024-03-14-blenheim-05"]
file_lists = ["images.zip", "lidar_slam.zip", "T_gt_lidar.txt"]
for sequence in dataset_sequences:
    output_folder = Path(__file__).parent.parent / "data"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {sequence} to {output_folder}")
    for file in file_lists:
        hf_hub_download(repo_id, repo_type="dataset", filename=file, subfolder=sequence, local_dir=str(output_folder))
    files = list(output_folder.rglob("*.zip"))
    for file in files:
        logger.info(f"Unzipping {file}")
        shutil.unpack_archive(file, file.parent)
        file.unlink()
    logger.info(f'Complete downloading "{sequence}" to {output_folder}')
