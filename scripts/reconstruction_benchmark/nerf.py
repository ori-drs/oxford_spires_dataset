import sys
from pathlib import Path

from nerfstudio.scripts.exporter import entrypoint as exporter_entrypoint
from nerfstudio.scripts.train import entrypoint as train_entrypoint

from oxford_spires_utils.bash_command import print_with_colour


def generate_nerfstudio_config(
    method, data_dir, output_dir, iterations=30000, eval_step=500, vis="wandb", cam_opt_mode="off"
):
    ns_config = {
        "method": method,
        "data": str(data_dir),
        "output-dir": str(output_dir),
        "vis": vis,
        "max-num-iterations": iterations,
        "pipeline.model.camera-optimizer.mode": cam_opt_mode,
        "steps-per-eval-image": eval_step,
    }
    return ns_config


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


def update_argv(nerfstudio_config):
    assert sys.argv[0].endswith(".py") and len(sys.argv) == 1, "No args should be provided for the script"
    for k, v in nerfstudio_config.items():
        if k == "method":
            sys.argv.append(f"{v}")
        else:
            sys.argv.append(f"--{k}")
            sys.argv.append(f"{v}")
    print_with_colour(" ".join(sys.argv))


def run_nerfstudio(ns_config):
    update_argv(ns_config)
    train_entrypoint()
    sys.argv = [sys.argv[0]]
    output_log_dir = Path(ns_config["output-dir"]) / Path(ns_config["data"]).name / ns_config["method"]
    lastest_output_folder = sorted([x for x in output_log_dir.glob("*") if x.is_dir()])[-1]
    latest_output_config = lastest_output_folder / "config.yml"
    export_method = "gaussian-splat" if ns_config["method"] == "splatfacto" else "pointcloud"
    run_nerfstudio_exporter(latest_output_config, export_method)


def run_nerfstudio_exporter(config_file, export_method):
    exporter_config = {
        "method": export_method,
        "load-config": config_file,
        "output-dir": config_file.parent,
    }
    if export_method == "pointcloud":
        exporter_config["normal-method"] = "open3d"
    if export_method == "gaussian-splat":
        exporter_config["ply-color-mode"] = "rgb"
    update_argv(exporter_config)
    exporter_entrypoint()
    sys.argv = [sys.argv[0]]
