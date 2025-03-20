import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download, list_repo_files


class DatasetDownloader:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.repo_id = "ori-drs/oxford_spires_dataset"
        self.ground_truth_sites = [
            "blenheim-palace",
            "christ-church",
            "keble-college",
            "observatory-quarter",
        ]
        self.ground_truth_lists = [f"ground_truth_cloud/{site}" for site in self.ground_truth_sites]
        self.cloud_file = "individual_cloud_e57.zip"

        # Cache for file lists
        self.sequence_files: Dict[str, List[str]] = {}
        self._dataset_sequences: List[str] = []
        self.local_sequences: List[str] = []  # New attribute for local sequences
        self.load_remote_files()
        self.load_remote_sequences()
        self.load_local_sequences()

    def load_remote_files(self):
        try:
            files = list_repo_files(self.repo_id, repo_type="dataset")
            self.remote_files = sorted(files)
            self.logger.info(f"Loaded {len(self.remote_files)} files")
        except Exception as e:
            self.logger.error(f"Failed to load files: {str(e)}")

    def load_remote_sequences(self):
        """Load available sequences from Hugging Face."""
        try:
            # Look for directories that match our sequence pattern (YYYY-MM-DD-*-*)
            sequence_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-.*$")
            sequences = set()
            for file in self.remote_files:
                parts = file.split("/")
                if len(parts) > 1 and sequence_pattern.match(parts[0]):
                    sequences.add(parts[0])

            self.remote_sequences = sorted(list(sequences))
            self.logger.info(f"Loaded {len(self.remote_sequences)} sequences")
        except Exception as e:
            self.logger.error(f"Failed to load sequences: {str(e)}")

    def load_local_sequences(self):
        """Load local sequences from the base directory."""
        base_path = Path(self.base_dir)
        if not base_path.exists():
            self.local_sequences = []
            return

        self.local_sequences = [
            d.name for d in base_path.iterdir() if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}-.*$", d.name)
        ]
        self.logger.info(f"Loaded {len(self.local_sequences)} local sequences")

    def load_remote_sequence_files(self, sequence_name: str):
        """Load available files for a specific sequence from Hugging Face."""
        try:
            files = list_repo_files(self.repo_id, repo_type="dataset", subfolder=f"{sequence_name}/raw")
            self.sequence_files[sequence_name] = files
            self.logger.info(f"Loaded {len(files)} files for sequence {sequence_name}")
        except Exception as e:
            self.logger.error(f"Failed to load files for sequence {sequence_name}: {str(e)}")

    def download_subfolder(self, subfolder: str, progress_callback=None) -> bool:
        """
        Download a specific sequence from the dataset.

        Args:
            sequence_name: Name of the sequence to download
            progress_callback: Optional callback function to report download progress

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            # check if any remote files match the subfolder
            remote_subfolder_files = [f for f in self.remote_files if f.startswith(subfolder)]
            if not remote_subfolder_files:
                self.logger.error(f"Invalid subfolder: {sequence_name}")
                return False

            output_folder = self.base_dir
            output_folder.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Downloading {subfolder} to {output_folder}")

            # # Get the remote file list

            for file in remote_subfolder_files:
                hf_hub_download(
                    self.repo_id,
                    repo_type="dataset",
                    filename=file,
                    # subfolder=subfolder,
                    local_dir=str(output_folder),
                )

            # Unzip downloaded files
            files = list(output_folder.rglob("*.zip"))
            for file in files:
                self.logger.info(f"Unzipping {file}")
                shutil.unpack_archive(file, file.parent)
                file.unlink()

            self.logger.info(f'Complete downloading "{subfolder}" to {output_folder}')
            return True

        except Exception as e:
            self.logger.error(f"Failed to download sequence {subfolder}: {str(e)}")
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


def main():
    base_dir = "/home/docker_dev/data"
    downloader = DatasetDownloader(base_dir)
    downloader.download_subfolder("2024-03-12-keble-college-04/processed/trajectory")
    print(downloader.local_sequences)


if __name__ == "__main__":
    main()
