from copy import deepcopy
from pathlib import Path

from oxford_spires_utils.trajectory.nerf_json_handler import NeRFJsonHandler


def merge_two_json(json1, json2):
    assert json1.traj.keys() == json2.traj.keys()
    new_json = deepcopy(json1)
    new_json.traj["frames"] += json2.traj["frames"]
    return new_json


image_dir = "/home/docker_dev/oxford_spires_dataset/data/roq-full/images"
json_train_file = "/home/docker_dev/oxford_spires_dataset/data/roq-full/outputs/colmap/seq_1_fountain.json"
json_eval_file = "/home/docker_dev/oxford_spires_dataset/data/roq-full/outputs/colmap/seq_1_fountain_back.json"
new_image_dir = Path(image_dir).parent.resolve() / "images_train_eval"
merged_json_file = Path(json_train_file).parent / "merged.json"


json_train = NeRFJsonHandler(json_train_file)
json_eval = NeRFJsonHandler(json_eval_file)
json_train.rename_filename(
    old_folder="images", new_folder=new_image_dir.stem, prefix="train_", base_folder=str(Path(image_dir).parent)
)
json_eval.rename_filename(
    old_folder="images", new_folder=new_image_dir.stem, prefix="eval_", base_folder=str(Path(image_dir).parent)
)


# merge
new_json = merge_two_json(json_train, json_eval)
new_json.save_json(str(merged_json_file))

# create json with the new train/eval prefix
