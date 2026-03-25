import argparse
import logging
import shutil
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from oxspires_tools.dataset import OxfordSpiresDataset
from oxspires_tools.utils import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_patterns(
    repo_id: str,
    patterns: list,
    local_dir: str,
    repo_type: str = "dataset",
    branch: str = "main",
    unpack: bool = False,
) -> list:
    """Download patterns from the HuggingFace repository."""
    logger.info(f"Repository: {repo_id}")
    logger.info(f"Local directory: {local_dir}")
    logger.info(f"Unpack archives: {unpack}")
    logger.info(f"Downloading {len(patterns)} pattern(s)...\n")

    for i, pattern in enumerate(patterns, 1):
        logger.info(f"🚀 [{i}/{len(patterns)}] {pattern}")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=pattern,
            local_dir=local_dir,
            repo_type=repo_type,
            use_auth_token=False,
            revision=branch,
        )
        logger.info(f"✅ Downloaded: {pattern}\n")
    logger.info("🏁 All downloads complete!")


def unpack_zip_files(local_dir: str):
    zip_files = list(Path(local_dir).rglob("*.zip"))
    for zip_file in zip_files:
        logger.info(f"Unzipping {zip_file}")
        shutil.unpack_archive(zip_file, extract_dir=zip_file.parent)
        zip_file.unlink()
    logger.info("🏁 All zip files unpacked!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_download.yaml",
        help="Path to configuration file (default: config/dataset_download.yaml)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=None,
        help="Override patterns to download (e.g. 'sequences/foo/*')",
    )
    args = parser.parse_args()
    return args


def main():
    _ = setup_logging()
    args = get_args()

    config = load_config(args.config)
    Path(config["local_dir"]).mkdir(parents=True, exist_ok=True)

    patterns = args.patterns if args.patterns is not None else config["patterns"]
    if config.get("download", True):
        download_patterns(
            repo_id=config["repo_id"],
            patterns=patterns,
            local_dir=config["local_dir"],
            repo_type=config["repo_type"],
            branch=config["branch"],
        )
    if config.get("unpack", True):
        unpack_zip_files(config["local_dir"])
    if config.get("check", True):
        sequences_dir = Path(config["local_dir"]) / "sequences"
        sequences_list = sorted([d.name for d in sequences_dir.iterdir() if d.is_dir()])
        for i, seq_name in enumerate(sequences_list):
            seq_dir = sequences_dir / seq_name
            if not seq_dir.is_dir():
                continue
            logger.info(f"\n🚀 [{i}/{len(sequences_list)}] Checking sequence: {seq_dir.name}")
            dataset = OxfordSpiresDataset(seq_dir)
            dataset.check_image_lidar_sync(cam_id=0, tolerance_sec=0.0)


if __name__ == "__main__":
    main()
