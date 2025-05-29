import json
import shutil
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_command(cmd, log_path=None):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
    for line in process.stdout:
        print(line, end="")
        if log_path is not None:
            assert isinstance(log_path, (Path, str))
            with open(log_path, "a") as f:
                f.write(line)


def run_cmvs(
    colmap_dir, cmvs_pmvs_exe_folder, final_json_path, image_dir=None, max_submap_image=400, plt_pose_size=0.4
):
    colmap_dir = Path(colmap_dir)
    colmap_sparse_0 = colmap_dir / "sparse/0"
    dense_folder = colmap_dir / "dense"
    cmvs_folder = dense_folder / "pmvs"
    submap_plt_path = colmap_dir / "submap_cmvs" / "submap_poses.png"
    submap_json_folder = colmap_dir / "submap_cmvs"
    cmvs_id_file_path = cmvs_folder / "bundle.rd.out.list.txt"

    if image_dir is None:
        image_dir = colmap_dir
    image_dir = Path(image_dir)

    run_submap_visualize = False
    run_submap_json_images_test = False

    if dense_folder.exists():
        shutil.rmtree(dense_folder)
    if submap_json_folder.exists():
        shutil.rmtree(submap_json_folder)
    submap_json_folder.mkdir(exist_ok=True, parents=True)
    assert (image_dir / "images").exists(), f"{image_dir / 'images'} does not exist"
    undistorter_cmd = [
        "colmap image_undistorter",
        f"--image_path {image_dir}",
        f"--input_path {colmap_sparse_0}",
        f"--output_path {dense_folder}",
        "--output_type PMVS",
    ]
    undistorter_cmd = " ".join(undistorter_cmd)
    run_command(undistorter_cmd)

    if Path(cmvs_folder).joinpath("centers-all.ply").exists():
        [p.unlink(missing_ok=True) for p in Path(cmvs_folder).glob("centers-*")]
    if Path(cmvs_folder).joinpath("option-0001").exists():
        [p.unlink(missing_ok=True) for p in Path(cmvs_folder).glob("option-*")]

    with open(cmvs_id_file_path, "r") as f:
        cmvs_image_paths = f.readlines()

    cmd_cmvs = f"{cmvs_pmvs_exe_folder}/cmvs {cmvs_folder}/ {max_submap_image}"
    run_command(cmd_cmvs)

    cmd_cmvs_options = f"{cmvs_pmvs_exe_folder}/genOption {cmvs_folder}/"
    run_command(cmd_cmvs_options)

    options_file_list = sorted(Path(cmvs_folder).glob("option-*"))
    # remove option-all
    options_file_list = [str(f) for f in options_file_list if "option-all" not in str(f)]

    clusters = {}
    num_images = {}
    for option_file in options_file_list:
        print(f"Processing {option_file}")
        with open(option_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("timages"):
                timages = line.split()[1:]
                break
        timages = [int(i) for i in timages]
        cluster_images = timages[1:]
        num_images_current = len(cluster_images)
        assert len(cluster_images) == timages[0]
        clusters[option_file] = cluster_images
        num_images[option_file] = num_images_current

    unique_images = set()
    for cluster in clusters.values():
        unique_images.update(cluster)
    print(f"Unique Images {len(unique_images)}")
    print(f"Total Images {sum(num_images.values())}")

    cmvs_frames = []
    cmvs_poses = []
    cmvs_indices = []

    with open(final_json_path, "r") as f:
        transform_data = json.load(f)
    k = -1
    mismatch_printed = False
    for pose in transform_data["frames"]:
        k += 1
        pose_file_path_query = f"{pose['file_path']}\n"
        image_id = cmvs_image_paths.index(pose_file_path_query) if pose_file_path_query in cmvs_image_paths else None
        if k != image_id and not mismatch_printed:
            print(
                f"Image id mismatch: {k} {image_id}, so pose file is not ordered as cmvs image paths. Only happens when json drops frames that has no lidar"
            )
            print("This will only be printed once")
            mismatch_printed = True
        if image_id is None:
            continue
        assert isinstance(image_id, int)
        cluster_ids = [
            i for i, (cluster_id, cluster_images) in enumerate(clusters.items()) if image_id in cluster_images
        ]

        if len(cluster_ids) == 0:
            continue
        if len(cluster_ids) > 1:
            print(f"Multiple cluster ids: {cluster_ids} for image {image_id} {pose['file_path']}")

        xyz = np.asarray(pose["transform_matrix"])[:3, 3]
        for cluster_id in cluster_ids:
            cmvs_frames.append(pose)
            cmvs_poses.append(xyz)
            cmvs_indices.append(cluster_id)

    colours = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    cmvs_poses = np.array(cmvs_poses)
    cmvs_indices = np.array(cmvs_indices)
    num_submaps = len(np.unique(cmvs_indices))
    submap_image_num = {i: 0 for i in range(num_submaps)}
    for i in range(num_submaps):
        colour = colours[i % len(colours)]
        cluster_poses = cmvs_poses[cmvs_indices == i]
        plt.scatter(cluster_poses[:, 0], cluster_poses[:, 1], s=plt_pose_size, color=colour, label=f"submap_{i}")

        # save json
        json_template = deepcopy(transform_data)
        submap_frames = [cmvs_frames[j] for j in range(len(cmvs_frames)) if cmvs_indices[j] == i]
        json_template["frames"] = submap_frames
        submap_json_path = submap_json_folder / f"submap_{i}.json"
        with open(submap_json_path, "w") as f:
            json.dump(json_template, f, indent=4)

        image_list = [cmvs_frame["file_path"] for cmvs_frame in submap_frames]
        if run_submap_json_images_test:
            for image in image_list:
                full_path = image_dir / image
                assert full_path.exists(), f"{full_path} does not exist"
                new_path = dense_folder / "submap_json_images_test" / f"submap{i}" / image
                submap_image_num[i] += 1
                new_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(full_path, new_path)
    for i in range(num_submaps):
        if submap_image_num[i] == len(clusters[options_file_list[i]]):
            continue
        else:
            print(f"submap {i} has {submap_image_num[i]} images but cluster has {len(clusters[options_file_list[i]])}")
    plt.legend()
    plt.savefig(submap_plt_path)

    xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
    for i in range(num_submaps):
        colour = colours[i % len(colours)]
        cluster_poses = cmvs_poses[cmvs_indices == i]
        plt.figure()
        plt.scatter(cluster_poses[:, 0], cluster_poses[:, 1], s=plt_pose_size, color=colour, label=f"submap_{i}")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.savefig(submap_plt_path.parent / f"submap_{i}.png")

    if run_submap_visualize:
        image_folder = Path(cmvs_folder) / "visualize"
        submap_image_folder = Path(cmvs_folder) / "submap_visualize"
        for option_file, cluster_images in clusters.items():
            print(f"Writing images for {option_file}")
            cluster_image_folder = submap_image_folder / Path(option_file).stem
            # mkdir but remove old one
            if cluster_image_folder.exists():
                shutil.rmtree(cluster_image_folder)
            cluster_image_folder.mkdir(exist_ok=True, parents=True)
            for image in cluster_images:
                image_path = image_folder / f"{image:0>8}.jpg"
                image_path_new = cluster_image_folder / f"{image:0>8}.jpg"
                shutil.copy(image_path, image_path_new)


if __name__ == "__main__":
    colmap_dir = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/processed/output_colmap"
    cmvs_pmvs_exe_folder = "/home/yifu/workspace/CMVS-PMVS/program/OutputLinux/main"
    image_dir = "/home/yifu/data/nerf_data_pipeline/2024-03-13-maths_1/raw"
    run_cmvs(colmap_dir, cmvs_pmvs_exe_folder, image_dir=image_dir)
