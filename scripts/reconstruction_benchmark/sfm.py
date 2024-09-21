import json
from pathlib import Path

import numpy as np
import requests
from nerfstudio.process_data.colmap_utils import colmap_to_json
from tqdm import tqdm

from oxford_spires_utils.bash_command import run_command
from oxford_spires_utils.se3 import s_se3_from_sim3

camera_model_list = {"OPENCV_FISHEYE", "OPENCV", "PINHOLE"}


def get_vocab_tree(image_num) -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    if image_num < 1000:
        vocab_tree_filename = Path(__file__).parent / "vocab_tree_flickr100K_words32K.bin"
        url = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
    elif image_num < 10000:
        vocab_tree_filename = Path(__file__).parent / "vocab_tree_flickr100K_words256K.bin"
        url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
    else:
        vocab_tree_filename = Path(__file__).parent / "vocab_tree_flickr100K_words1M.bin"
        url = "https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin"

    if not vocab_tree_filename.exists():
        response = requests.get(url, stream=True)
        with open(vocab_tree_filename, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=1024), unit="kB"):
                handle.write(data)
    return vocab_tree_filename


def run_colmap(image_path, output_path, camera_model="OPENCV_FISHEYE"):
    assert camera_model in camera_model_list, f"{camera_model} not supported. Supported models: {camera_model_list}"
    database_path = output_path / "database.db"
    sparse_path = output_path / "sparse"
    sparse_0_path = sparse_path / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)

    colmap_feature_extractor_cmd = [
        "colmap feature_extractor",
        f"--image_path {image_path}",
        f"--database_path {database_path}",
        "--ImageReader.single_camera_per_folder 1",
        f"--ImageReader.camera_model {camera_model}",
    ]
    colmap_feature_extractor_cmd = " ".join(colmap_feature_extractor_cmd)
    run_command(colmap_feature_extractor_cmd, print_command=True)

    image_num = len(list(image_path.rglob("*")))
    colmap_vocab_tree_matcher_cmd = [
        "colmap vocab_tree_matcher",
        f"--database_path {database_path}",
        f"--VocabTreeMatching.vocab_tree_path {get_vocab_tree(image_num)}",
    ]
    colmap_vocab_tree_matcher_cmd = " ".join(colmap_vocab_tree_matcher_cmd)
    run_command(colmap_vocab_tree_matcher_cmd, print_command=True)

    colmap_vocab_tree_matcher_cmd = " ".join(colmap_vocab_tree_matcher_cmd)
    colmap_mapper_cmd = [
        "colmap mapper",
        f"--database_path {database_path}",
        f"--image_path {image_path}",
        f"--output_path {sparse_path}",
    ]
    colmap_mapper_cmd = " ".join(colmap_mapper_cmd)
    run_command(colmap_mapper_cmd, print_command=True)

    colmap_ba_cmd = [
        "colmap bundle_adjuster",
        f"--input_path {sparse_0_path}",
        f"--output_path {sparse_0_path}",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    colmap_ba_cmd = " ".join(colmap_ba_cmd)
    run_command(colmap_ba_cmd, print_command=True)

    num_image_matched = colmap_to_json(recon_dir=sparse_0_path, output_dir=output_path)
    print(f"Number of images matched: {num_image_matched}")


def rescale_colmap_json(json_file, sim3_matrix, output_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    new_frames = []
    for frame in data["frames"]:
        T = np.array(frame["transform_matrix"])
        T_test = sim3_matrix @ T
        scale, T_vilens_colmap = s_se3_from_sim3(sim3_matrix)
        T[:3, 3] *= scale
        T = T_vilens_colmap @ T

        T_test[:3, :3] /= scale
        assert np.allclose(T, T_test)

        frame["transform_matrix"] = T.tolist()
        new_frames.append(frame)
    data["frames"] = new_frames

    assert Path(output_file).suffix == ".json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
