import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from nerfstudio.scripts.eval import entrypoint as eval_entrypoint
from nerfstudio.scripts.exporter import entrypoint as exporter_entrypoint
from nerfstudio.scripts.train import entrypoint as train_entrypoint

from oxford_spires_utils.bash_command import print_with_colour

logger = logging.getLogger(__name__)


def generate_nerfstudio_config(
    method, data_dir, output_dir, iterations=5000, eval_step=500, vis="wandb", cam_opt_mode="off", eval_mode="fraction"
):
    exp_name = Path(data_dir).name if Path(data_dir).is_dir() else Path(data_dir).parent.name
    ns_config = {
        "method": method,
        "experiment-name": str(exp_name),
        "output-dir": str(output_dir),
        "vis": vis,
        "max-num-iterations": iterations,
        "pipeline.model.camera-optimizer.mode": cam_opt_mode,
        "steps-per-eval-image": eval_step,
    }
    ns_data_config = {
        "dataparser": "nerfstudio-data",
        "data": str(data_dir),
        "eval-mode": eval_mode,
    }
    return ns_config, ns_data_config


def create_nerfstudio_dir(colmap_dir, ns_dir, image_dir):
    ns_dir = Path(ns_dir).resolve()
    colmap_dir = Path(colmap_dir).resolve()
    image_dir = Path(image_dir).resolve()
    # Ensure ns_dir exists
    ns_dir.mkdir(parents=True, exist_ok=True)

    # Symlink image_dir to ns_dir
    image_symlink = ns_dir / image_dir.name
    if not image_symlink.exists():
        relative_image_dir = Path(os.path.relpath(str(image_dir.parent), str(ns_dir))) / image_dir.name
        image_symlink.symlink_to(relative_image_dir)

    # Symlink contents of colmap_dir to ns_dir
    for item in colmap_dir.iterdir():
        item_symlink = ns_dir / item.name
        if not item_symlink.exists():
            relative_item = Path(os.path.relpath(str(colmap_dir), str(ns_dir))) / item.name
            item_symlink.symlink_to(relative_item)


def update_argv(nerfstudio_config, follow_up=False):
    if not follow_up:
        assert sys.argv[0].endswith(".py") and len(sys.argv) == 1, "No args should be provided for the script"
    for k, v in nerfstudio_config.items():
        if k in ("method", "dataparser"):
            sys.argv.append(f"{v}")
        else:
            sys.argv.append(f"--{k}")
            sys.argv.append(f"{v}")
    print_with_colour(" ".join(sys.argv))


def run_nerfstudio(ns_config, ns_data_config):
    logger.info(f"Running '{ns_config['method']}' on {ns_data_config['data']}")
    logging.disable(logging.DEBUG)
    update_argv(ns_config)
    update_argv(ns_data_config, follow_up=True)
    train_entrypoint()
    sys.argv = [sys.argv[0]]
    folder_name = ns_config["experiment-name"]
    # rename nerfacto-big or nerfacto-huge to nerfacto, splatfacto-big to splatfacto
    method_dir_name = ns_config["method"].replace("-big", "").replace("-huge", "")
    output_log_dir = Path(ns_config["output-dir"]) / folder_name / method_dir_name
    lastest_output_folder = sorted([x for x in output_log_dir.glob("*") if x.is_dir()])[-1]
    latest_output_config = lastest_output_folder / "config.yml"

    # evaluate renders
    logger.info(f"Evaluating from {lastest_output_folder}")
    render_dir = lastest_output_folder / "renders"
    run_nerfstudio_eval(latest_output_config, render_dir)
    logging.disable(logging.NOTSET)

    # export cloud
    export_method = "gaussian-splat" if ns_config["method"] == "splatfacto" else "pointcloud"
    output_cloud_file = run_nerfstudio_exporter(latest_output_config, export_method)
    ns_se3, scale_matrix = load_ns_transform(lastest_output_folder)
    cloud = o3d.io.read_point_cloud(str(output_cloud_file))
    cloud.transform(scale_matrix)
    cloud.transform(np.linalg.inv(ns_se3))
    final_metric_cloud_file = output_cloud_file.with_name(f'{ns_config["method"]}_cloud_metric.ply')
    o3d.io.write_point_cloud(str(final_metric_cloud_file), cloud)
    return final_metric_cloud_file


def run_nerfstudio_eval(config_file, render_dir):
    output_eval_file = config_file.parent / "eval_results.json"
    eval_config = {
        "load-config": config_file,
        "output-path": output_eval_file,
        "render-output-path": render_dir,
    }
    update_argv(eval_config)
    eval_entrypoint()
    logger.info(f"Nerfstudio eval results\n{json.load(output_eval_file.open())}")
    sys.argv = [sys.argv[0]]


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
