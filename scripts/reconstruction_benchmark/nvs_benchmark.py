import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from nerf import generate_nerfstudio_config, run_nerfstudio


def run_ns(
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


def unzip_files(zip_files):
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(zip_file.parent)
            print(f"Extracted {zip_file} to {zip_file.parent}")
    return zip_files


def run_nvs_benchmark(undistorted_ns_dir):
    run_ns("nerfacto", undistorted_ns_dir, "transforms_train.json", "fraction")
    run_ns("nerfacto", undistorted_ns_dir, "transforms_train_eval.json", "filename")
    run_ns("nerfacto-big", undistorted_ns_dir, "transforms_train.json", "fraction")
    run_ns("nerfacto-big", undistorted_ns_dir, "transforms_train_eval.json", "filename")
    run_ns("splatfacto", undistorted_ns_dir, "transforms_train.json", "fraction")
    run_ns("splatfacto", undistorted_ns_dir, "transforms_train_eval.json", "filename")
    run_ns("splatfacto-big", undistorted_ns_dir, "transforms_train.json", "fraction")
    run_ns("splatfacto-big", undistorted_ns_dir, "transforms_train_eval.json", "filename")


if __name__ == "__main__":
    hf_repo_id = "ori-drs/oxford_spires_dataset"
    example_pattern = (
        "novel_view_synthesis_benchmark/training_data/nvs*"  # download the whole novel view synthesis benchmark
    )
    local_dir = "download"
    snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=example_pattern,
        local_dir=local_dir,
        repo_type="dataset",
        use_auth_token=False,
    )

    # unzip the downloaded files
    zip_file_dir = Path(__file__).parent.parent.parent / local_dir
    zip_files = list(zip_file_dir.rglob("*.zip"))
    unzip_files(zip_files)

    for zip_file in zip_files:
        undistorted_ns_dir = zip_file.parent / zip_file.stem
        print(f"Running NVS benchmark for {undistorted_ns_dir}...")
        run_nvs_benchmark(undistorted_ns_dir)
