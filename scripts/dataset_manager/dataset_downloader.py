import logging
import shutil
from pathlib import Path
from typing import List, Optional

from huggingface_hub import hf_hub_download


class DatasetDownloader:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.repo_id = "ori-drs/oxford_spires_dataset"
        self.dataset_sequences = [
            "2024-03-12-keble-college-04",
            "2024-03-13-observatory-quarter-01",
            "2024-03-14-blenheim-palace-05",
            "2024-03-18-christ-church-02",
        ]
        # self.file_lists = ["images.zip", "lidar_slam.zip", "T_gt_lidar.txt"]
        self.file_lists = ["imu.csv"]
        self.ground_truth_sites = [
            "blenheim-palace",
            "christ-church",
            "keble-college",
            "observatory-quarter",
        ]
        self.ground_truth_lists = [f"ground_truth_cloud/{site}" for site in self.ground_truth_sites]
        self.cloud_file = "individual_cloud_e57.zip"

    def download_sequence(self, sequence_name: str, progress_callback=None) -> bool:
        """
        Download a specific sequence from the dataset.

        Args:
            sequence_name: Name of the sequence to download
            progress_callback: Optional callback function to report download progress

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            if sequence_name not in self.dataset_sequences:
                self.logger.error(f"Invalid sequence name: {sequence_name}")
                return False

            output_folder = self.base_dir
            output_folder.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Downloading {sequence_name} to {output_folder}")

            for file in self.file_lists:
                hf_hub_download(
                    self.repo_id,
                    repo_type="dataset",
                    filename=file,
                    subfolder=f"{sequence_name}/raw",
                    local_dir=str(output_folder),
                )

            # Unzip downloaded files
            files = list(output_folder.rglob("*.zip"))
            for file in files:
                self.logger.info(f"Unzipping {file}")
                shutil.unpack_archive(file, file.parent)
                file.unlink()

            self.logger.info(f'Complete downloading "{sequence_name}" to {output_folder}')
            return True

        except Exception as e:
            self.logger.error(f"Failed to download sequence {sequence_name}: {str(e)}")
            return False

    def download_ground_truth(self, site_name: str, progress_callback=None) -> bool:
        """
        Download ground truth data for a specific site.

        Args:
            site_name: Name of the site to download ground truth for
            progress_callback: Optional callback function to report download progress

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            if site_name not in self.ground_truth_sites:
                self.logger.error(f"Invalid site name: {site_name}")
                return False

            output_folder = self.base_dir
            output_folder.mkdir(parents=True, exist_ok=True)

            ground_truth_path = f"ground_truth_cloud/{site_name}"
            self.logger.info(f"Downloading {ground_truth_path} to {output_folder}")

            hf_hub_download(
                self.repo_id,
                repo_type="dataset",
                filename=self.cloud_file,
                subfolder=ground_truth_path,
                local_dir=str(output_folder),
            )

            # Unzip downloaded files
            files = list(output_folder.rglob("*.zip"))
            for file in files:
                self.logger.info(f"Unzipping {file}")
                shutil.unpack_archive(file, file.parent)
                file.unlink()

            self.logger.info(f'Complete downloading ground truth for "{site_name}" to {output_folder}')
            return True

        except Exception as e:
            self.logger.error(f"Failed to download ground truth for {site_name}: {str(e)}")
            return False

    def list_available_sequences(self) -> List[str]:
        """
        List all available sequences in the dataset.

        Returns:
            List[str]: List of sequence names
        """
        return self.dataset_sequences

    def list_available_sites(self) -> List[str]:
        """
        List all available sites for ground truth data.

        Returns:
            List[str]: List of site names
        """
        return self.ground_truth_sites

    def get_sequence_path(self, sequence_name: str) -> Optional[Path]:
        """
        Get the path to a specific sequence.

        Args:
            sequence_name: Name of the sequence

        Returns:
            Optional[Path]: Path to the sequence if it exists, None otherwise
        """
        sequence_path = self.base_dir / sequence_name
        return sequence_path if sequence_path.exists() else None

    def get_ground_truth_path(self, site_name: str) -> Optional[Path]:
        """
        Get the path to ground truth data for a specific site.

        Args:
            site_name: Name of the site

        Returns:
            Optional[Path]: Path to the ground truth data if it exists, None otherwise
        """
        ground_truth_path = self.base_dir / "ground_truth_cloud" / site_name
        return ground_truth_path if ground_truth_path.exists() else None
