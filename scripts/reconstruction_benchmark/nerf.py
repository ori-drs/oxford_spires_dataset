import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
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
    ns_data = Path(ns_config["data"])
    folder_name = ns_data.name if ns_data.is_dir() else ns_data.parent.name
    output_log_dir = Path(ns_config["output-dir"]) / folder_name / ns_config["method"]
    lastest_output_folder = sorted([x for x in output_log_dir.glob("*") if x.is_dir()])[-1]
    latest_output_config = lastest_output_folder / "config.yml"
    export_method = "gaussian-splat" if ns_config["method"] == "splatfacto" else "pointcloud"
    output_cloud_file = run_nerfstudio_exporter(latest_output_config, export_method)
    ns_se3, scale_matrix = load_ns_transform(lastest_output_folder)
    cloud = o3d.io.read_point_cloud(str(output_cloud_file))

    cloud.transform(scale_matrix)
    cloud.transform(np.linalg.inv(ns_se3))
    final_metric_cloud_file = f"{ns_config["method"]}_cloud_metric.ply"
    o3d.io.write_point_cloud(str(output_cloud_file.with_name(final_metric_cloud_file)), cloud)


def run_nerfstudio_exporter(config_file, export_method):
    exporter_config = {
        "method": export_method,
        "load-config": config_file,
        "output-dir": config_file.parent,
    }
    if export_method == "pointcloud":
        exporter_config["normal-method"] = "open3d"
        # exporter_config["save-world-frame"] = True
        output_cloud_name = "point_cloud.ply"
    if export_method == "gaussian-splat":
        exporter_config["ply-color-mode"] = "rgb"
        output_cloud_name = "splat.ply"
    update_argv(exporter_config)
    exporter_entrypoint()
    sys.argv = [sys.argv[0]]
    output_cloud_file = exporter_config["output-dir"] / output_cloud_name
    return output_cloud_file


def load_ns_transform(ns_log_output_dir):
    transform_json_file = ns_log_output_dir / "dataparser_transforms.json"
    transform_data = json.load(transform_json_file.open())
    se3_matrix = np.array(transform_data["transform"])
    se3_matrix = np.vstack([se3_matrix, [0, 0, 0, 1]])
    scale = transform_data["scale"]
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= 1 / scale
    return se3_matrix, scale_matrix
