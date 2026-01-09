import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_patterns(repo_id: str, patterns: list, local_dir: str, repo_type: str = "dataset") -> None:
    """Download patterns from the HuggingFace repository."""
    print(f"Repository: {repo_id}")
    print(f"Local directory: {local_dir}")
    print(f"Downloading {len(patterns)} pattern(s)...\n")

    for i, pattern in enumerate(patterns, 1):
        print(f"[{i}/{len(patterns)}] {pattern}")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=pattern,
            local_dir=local_dir,
            repo_type=repo_type,
            use_auth_token=False,
        )
        print(f"✅ Downloaded: {pattern}\n")

    print("🏁 All downloads complete!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_download.yaml",
        help="Path to configuration file (default: config/dataset_download.yaml)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        nargs="+",
        help="Pattern(s) to download (overrides config patterns)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory to download to (overrides config)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID (overrides config)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    config = load_config(args.config)

    repo_id = args.repo_id or config["repo_id"]
    local_dir = args.local_dir or config["local_dir"]
    repo_type = config["repo_type"]
    patterns = args.pattern or config["patterns"]

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    download_patterns(repo_id, patterns, local_dir, repo_type)


if __name__ == "__main__":
    main()
