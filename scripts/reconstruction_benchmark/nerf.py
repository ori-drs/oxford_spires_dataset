from pathlib import Path

from oxford_spires_utils.bash_command import run_command


def create_nerfstudio_dir(colmap_dir, ns_dir, image_dir):
    ns_dir = Path(ns_dir)
    colmap_dir = Path(colmap_dir)
    image_dir = Path(image_dir)
    # Ensure ns_dir exists
    ns_dir.mkdir(parents=True, exist_ok=True)

    # Symlink image_dir to ns_dir
    image_symlink = ns_dir / image_dir.name
    if not image_symlink.exists():
        image_symlink.symlink_to(image_dir)

    # Symlink contents of colmap_dir to ns_dir
    for item in colmap_dir.iterdir():
        item_symlink = ns_dir / item.name
        if not item_symlink.exists():
            item_symlink.symlink_to(item)


def run_nerfacto(data_path, vis="wandb", cam_opt_mode="off"):
    assert Path(data_path).exists(), f"Data path not found at {data_path}"
    nerfstudio_cmd = [
        "ns-train",
        "nerfacto",
        f"--data {str(data_path)}",
        f"--vis {vis}",
        f"--pipeline.model.camera-optimizer.mode {cam_opt_mode}",
    ]
    nerfstudio_cmd = " ".join(nerfstudio_cmd)
    run_command(nerfstudio_cmd, print_command=True)


def run_splatfacto(data_path, vis="wandb", cam_opt_mode="off"):
    assert Path(data_path).exists(), f"Data path not found at {data_path}"
    nerfstudio_cmd = [
        "ns-train",
        "splatfacto",
        f"--data {str(data_path)}",
        f"--vis {vis}",
        f"--pipeline.model.camera-optimizer.mode {cam_opt_mode}",
    ]
    nerfstudio_cmd = " ".join(nerfstudio_cmd)
    run_command(nerfstudio_cmd, print_command=True)
