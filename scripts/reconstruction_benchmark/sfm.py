import json
from pathlib import Path

import numpy as np
import requests
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary
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

    # from nerfstudio.process_data.colmap_utils import colmap_to_json
    # num_image_matched = colmap_to_json(recon_dir=sparse_0_path, output_dir=output_path)
    export_json(sparse_0_path, json_file_name="transforms.json", output_dir=output_path, camera_model=camera_model)
    # print(f"Number of images matched: {num_image_matched}")


def rescale_colmap_json(json_file, sim3_matrix, output_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    new_frames = []
    for frame in data["frames"]:
        T = np.array(frame["transform_matrix"])
        # T_test = sim3_matrix @ T
        scale, T_vilens_colmap = s_se3_from_sim3(sim3_matrix)
        T[:3, 3] *= scale
        T = T_vilens_colmap @ T

        # T_test[:3, :3] /= scale
        # assert np.allclose(T, T_test)

        frame["transform_matrix"] = T.tolist()
        new_frames.append(frame)
    data["frames"] = new_frames

    assert Path(output_file).suffix == ".json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def export_json(input_bin_dir=None, json_file_name="transforms.json", output_dir=None, camera_model="OPENCV_FISHEYE"):
    camera_mask_path = None
    input_bin_dir = Path(input_bin_dir)
    cameras_path = input_bin_dir / "cameras.bin"
    images_path = input_bin_dir / "images.bin"
    output_dir = input_bin_dir if output_dir is None else Path(output_dir)

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)

    frames = []
    up = np.zeros(3)
    for _, im_data in images.items():
        camera = cameras[im_data.camera_id]
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)

        # https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py#L328
        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1  # flip whole world upside down
        up += c2w[0:3, 1]

        frame = generate_json_camera_data(camera, camera_model)
        frame["file_path"] = Path(f"./images/{im_data.name}").as_posix()  # assume images not in image path in colmap
        frame["transform_matrix"] = c2w.tolist()
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()

        frames.append(frame)

    out = {}
    out["camera_model"] = camera_model
    out["frames"] = frames
    num_frames_string = f"Number of frames: {len(frames)}"
    print(num_frames_string)

    # Save for scale adjustment later
    assert json_file_name[-5:] == ".json"
    with open(output_dir / json_file_name, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)


def generate_json_camera_data(camera, camera_model):
    assert camera_model in ["OPENCV_FISHEYE", "OPENCV"]
    data = {
        "fl_x": float(camera.params[0]),
        "fl_y": float(camera.params[1]),
        "cx": float(camera.params[2]),
        "cy": float(camera.params[3]),
        "w": camera.width,
        "h": camera.height,
    }

    if camera_model == "OPENCV":
        data.update(
            {
                "k1": float(camera.params[4]),
                "k2": float(camera.params[5]),
                "p1": float(camera.params[6]),
                "p2": float(camera.params[7]),
            }
        )
    if camera_model == "OPENCV_FISHEYE":
        data.update(
            {
                "k1": float(camera.params[4]),
                "k2": float(camera.params[5]),
                "k3": float(camera.params[6]),
                "k4": float(camera.params[7]),
            }
        )
    return data
