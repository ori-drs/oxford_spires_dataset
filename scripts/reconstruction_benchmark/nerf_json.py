from copy import deepcopy
from pathlib import Path

from nerf import create_nerfstudio_dir

from oxford_spires_utils.trajectory.file_interfaces.nerf import NeRFTrajReader
from oxford_spires_utils.trajectory.nerf_json_handler import NeRFJsonHandler
from oxford_spires_utils.trajectory.utils import pose_to_ply


def split_json(json_file, start_time, end_time, save_path):
    nerf_json_handler = NeRFJsonHandler(json_file)
    nerf_json_handler.keep_timestamp_only(start_time, end_time)
    # save_path = Path(json_file).parent / f"{new_name}.json"
    nerf_json_handler.save_json(save_path)

    nerf_traj = NeRFTrajReader(save_path)
    nerf_pose = nerf_traj.read_file()
    pose_ply_file = save_path.with_suffix(".ply")
    pose_to_ply(nerf_pose, pose_ply_file)


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


dataset_folder = "/home/docker_dev/oxford_spires_dataset/data/roq-full"

train_name = "seq_1_fountain"
train_start_time = 1710338123.042936934
train_end_time = 1710338186.342030327

eval_name = "seq_1_fountain_back"
eval_start_time = 1710338353.039451086
eval_end_time = 1710338386.638942551

use_undistorted_image = True

dataset_folder = Path(dataset_folder).resolve()
colmap_folder = dataset_folder / "outputs/colmap"
if use_undistorted_image:
    colmap_folder = colmap_folder / "dense"
json_file = colmap_folder / "transforms.json"

train_save_path = Path(json_file).parent / f"{train_name}.json"
eval_save_path = Path(json_file).parent / f"{eval_name}.json"
split_json(json_file, train_start_time, train_end_time, train_save_path)
split_json(json_file, eval_start_time, eval_end_time, eval_save_path)


image_dir = dataset_folder / "images" if not use_undistorted_image else colmap_folder / "images"
json_train_file = colmap_folder / (train_name + ".json")
json_eval_file = colmap_folder / (eval_name + ".json")
new_image_dir = image_dir.parent / "images_train_eval"
merged_json_file = colmap_folder / "transforms_train_eval.json"
merge_json_files(json_train_file, json_eval_file, image_dir, new_image_dir, merged_json_file)
# create json with the new train/eval prefix

ns_dir = dataset_folder / "outputs" / "nerfstudio" / (dataset_folder.stem + "_undistorted")
create_nerfstudio_dir(colmap_folder, ns_dir, new_image_dir)
