import os
from pathlib import Path


def get_common_directories():
    """Get a list of common directories to choose from."""
    common_dirs = [
        str(Path.home()),
        str(Path.home() / "data"),
        str(Path.home() / "workspace"),
    ]
    return [d for d in common_dirs if os.path.exists(d)]
