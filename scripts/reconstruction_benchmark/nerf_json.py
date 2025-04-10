from copy import deepcopy
from pathlib import Path

from nerf import create_nerfstudio_dir

from oxford_spires_utils.trajectory.file_interfaces.nerf import NeRFTrajReader
from oxford_spires_utils.trajectory.nerf_json_handler import NeRFJsonHandler
from oxford_spires_utils.trajectory.utils import pose_to_ply


def select_json_with_time_range(json_file, start_time, end_time, save_path):
    nerf_json_handler = NeRFJsonHandler(json_file)
    nerf_json_handler.sort_frames()
    nerf_json_handler.keep_timestamp_only(start_time, end_time)
    nerf_json_handler.save_json(save_path)

    nerf_traj = NeRFTrajReader(save_path)
    nerf_pose = nerf_traj.read_file()
    pose_ply_file = save_path.with_suffix(".ply")
    pose_to_ply(nerf_pose, pose_ply_file)


def select_json_with_folder(json_file, img_folder, save_path):
    nerf_json_handler = NeRFJsonHandler(json_file)
    nerf_json_handler.sort_frames()
    nerf_json_handler.sync_with_folder(img_folder)
    nerf_json_handler.save_json(save_path)


def split_json_every_n(json_file, n, save_path_removed, save_path_kept):
    nerf_json_handler = NeRFJsonHandler(json_file)
    nerf_json_handler.sort_frames()
    train_json = deepcopy(nerf_json_handler)
    removed_frames = nerf_json_handler.skip_frames(8, return_removed=True)
    train_json.traj["frames"] = removed_frames
    train_json.save_json(save_path_removed)
    nerf_json_handler.save_json(save_path_kept)


def merge_json_files(json_train_file, json_eval_file, image_dir, new_image_dir, merged_json_file):
    json_train = NeRFJsonHandler(json_train_file)
    json_eval = NeRFJsonHandler(json_eval_file)
    json_train.rename_filename(
        old_folder="images", new_folder=new_image_dir.stem, prefix="train_", base_folder=str(Path(image_dir).parent)
    )
    json_train.save_json(str(merged_json_file.with_name("transforms_train.json")))
    json_eval.rename_filename(
        old_folder="images", new_folder=new_image_dir.stem, prefix="eval_", base_folder=str(Path(image_dir).parent)
    )
    # merge
    assert json_train.traj.keys() == json_eval.traj.keys()
    new_json = deepcopy(json_train)
    new_json.traj["frames"] += json_eval.traj["frames"]
    new_json.save_json(str(merged_json_file))


# dataset_folder = "/home/docker_dev/oxford_spires_dataset/data/roq-full"

# train_name = "seq_1_fountain"
# train_start_time = 1710338123.042936934
# train_end_time = 1710338186.342030327

# eval_name = "seq_1_fountain_back"
# eval_start_time = 1710338353.039451086
# eval_end_time = 1710338386.638942551

dataset_folder = "/home/docker_dev/oxford_spires_dataset/data/2024-03-14-blenheim-palace-all"
train_name = "seq_5"
eval_name = "seq_1"

use_undistorted_image = True

dataset_folder = Path(dataset_folder).resolve()
colmap_folder = dataset_folder / "outputs/colmap"
if use_undistorted_image:
    colmap_folder = colmap_folder / "dense"
json_file = colmap_folder / "transforms.json"
train_image_folder = dataset_folder / "train_val_image" / "train"
eval_image_folder = dataset_folder / "train_val_image" / "eval"

assert (train_image_folder / "images").exists(), f"{train_image_folder / 'images'} does not exist"
assert (eval_image_folder / "images").exists(), f"{eval_image_folder / 'images'} does not exist"

train_save_path = Path(json_file).parent / f"{train_name}.json"
eval_save_path = Path(json_file).parent / f"{eval_name}.json"
select_json_with_folder(json_file, train_image_folder, train_save_path)
select_json_with_folder(json_file, eval_image_folder, eval_save_path)
# select_json_with_time_range(json_file, train_start_time, train_end_time, train_save_path)
# select_json_with_time_range(json_file, eval_start_time, eval_end_time, eval_save_path)


# split_json_every_n(train_save_path, n=8, save_path_kept = train_save_path, save_path_removed
image_dir = dataset_folder / "images" if not use_undistorted_image else colmap_folder / "images"
json_train_file = colmap_folder / (train_name + ".json")
json_eval_file = colmap_folder / (eval_name + ".json")
new_image_dir = image_dir.parent / "images_train_eval"
merged_json_file = colmap_folder / "transforms_train_eval.json"
merge_json_files(json_train_file, json_eval_file, image_dir, new_image_dir, merged_json_file)
# create json with the new train/eval prefix

ns_dir = dataset_folder / "outputs" / "nerfstudio" / (dataset_folder.stem + "_undistorted")
print(f"Creating NeRF Studio directory at {ns_dir}")
create_nerfstudio_dir(colmap_folder, ns_dir, new_image_dir)
