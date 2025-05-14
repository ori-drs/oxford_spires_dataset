from pathlib import Path

from nerf import generate_nerfstudio_config, run_nerfstudio


def run_nvs_benchmark(
    method="nerfacto",
    ns_data_dir=None,
    json_filename="transforms_metric.json",
    eval_mode="fraction",
):
    ns_data_dir = Path(ns_data_dir).resolve()
    assert ns_data_dir.exists(), f"nerfstudio directory not found at {ns_data_dir}"
    ns_model_dir = ns_data_dir / "trained_models"
    ns_config, ns_data_config = generate_nerfstudio_config(
        method, ns_data_dir / json_filename, ns_model_dir, eval_mode=eval_mode
    )
    _ = run_nerfstudio(ns_config, ns_data_config, export_cloud=False)


if __name__ == "__main__":
    undistorted_ns_dir = "/home/docker_dev/oxford_spires_dataset/oxford_spires_dataset_download/novel_view_synthesis_benchmark/training_data/blenheim"
    run_nvs_benchmark("nerfacto", "transforms_train.json", "fraction", undistorted_ns_dir)
    run_nvs_benchmark("nerfacto", "transforms_train_eval.json", "filename", undistorted_ns_dir)
    run_nvs_benchmark("nerfacto-big", "transforms_train.json", "fraction", undistorted_ns_dir)
    run_nvs_benchmark("nerfacto-big", "transforms_train_eval.json", "filename", undistorted_ns_dir)
    run_nvs_benchmark("splatfacto", "transforms_train.json", "fraction", undistorted_ns_dir)
    run_nvs_benchmark("splatfacto", "transforms_train_eval.json", "filename", undistorted_ns_dir)
    run_nvs_benchmark("splatfacto-big", "transforms_train.json", "fraction", undistorted_ns_dir)
    run_nvs_benchmark("splatfacto-big", "transforms_train_eval.json", "filename", undistorted_ns_dir)
