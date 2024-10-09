import json
import shutil
from pathlib import Path


class NeRFJsonHandler:
    # remember to create a copy of the json when removing,
    # otherwise frames will be skipped
    def __init__(self, input_json_path) -> None:
        self.load_json(input_json_path)

    def load_json(self, json_path):
        with open(json_path, "r") as f:
            self.traj = json.load(f)

    def save_json(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.traj, f, indent=4)

    def get_n_frames(self):
        return len(self.traj["frames"])

    def sync_with_folder(self, folder_path, valid_ext=".jpg"):
        # get all files in subfolder
        ref_files = list(Path(folder_path).glob("**/*" + valid_ext))
        print(f"{len(self.traj['frames'])} files in json")

        count = 0
        frames_copy = self.traj["frames"].copy()
        for frame in self.traj["frames"]:
            file_path = frame["file_path"]
            exist = [file_path for ref_file in ref_files if file_path == ref_file.__str__()[-len(file_path) :]]
            if len(exist) == 0:
                # print(f"file {file_path} not exist, remove it from json")
                frames_copy.remove(frame)
                count += 1
        print(f"{len(ref_files)} files in reference folder")
        print(f"removed {count} files, {len(frames_copy)} left")
        self.traj["frames"] = frames_copy

    def remove_folder(self, folder_path):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            file_path = Path(frame["file_path"])
            if file_path.parent == Path(folder_path):
                self.traj["frames"].remove(frame)
                print(f"removed {file_path} from json")
        print(f"filter_folder {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

    def keep_folder(self, folder_path):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            file_path = Path(frame["file_path"])
            if file_path.parent != Path(folder_path):
                self.traj["frames"].remove(frame)
                print(f"removed {file_path} from json")
        print(f"filter_folder {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

    def rename_filename(self, old_folder=None, new_folder=None, prefix="", suffix="", base_folder=None):
        for frame in self.traj["frames"]:
            file_path = Path(frame["file_path"])
            if old_folder is not None and new_folder is not None:
                assert str(file_path).startswith(old_folder), f"{file_path} does not start with {old_folder}"
                new_file_path = Path(str(file_path).replace(old_folder, new_folder))
            new_file_path = str(new_file_path.parent / (prefix + new_file_path.stem + suffix + new_file_path.suffix))
            frame["file_path"] = new_file_path
            if base_folder is not None:
                abs_old_file = Path(base_folder) / file_path
                assert abs_old_file.exists(), f"{abs_old_file} not exist"
                abs_new_file = Path(base_folder) / new_file_path
                abs_new_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(abs_old_file, abs_new_file)

    def keep_timestamp_only(self, start_time, end_time):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            file_path = Path(frame["file_path"])
            timestamp = float(file_path.stem)
            if timestamp < start_time or timestamp > end_time:
                self.traj["frames"].remove(frame)
                print(f"removed {file_path} from json")
        print(f"keep_timestamp_only {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

    def remove_timestamp(self, start_time, end_time):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            file_path = Path(frame["file_path"])
            timestamp = float(file_path.stem)
            if timestamp >= start_time and timestamp <= end_time:
                self.traj["frames"].remove(frame)
                print(f"removed {file_path} from json")
        print(f"remove_timestamp {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

    def skip_frames(self, skip):
        frames_copy = self.traj["frames"].copy()
        # sort
        # frames_copy.sort(key=lambda x: float(Path(x["file_path"]).stem))

        self.traj["frames"] = frames_copy[::skip]
        print(f"Skipping: {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

    def remove_intrinsics(self):
        # frames_copy = self.traj["frames"].copy()
        for frame in self.traj["frames"]:
            frame.pop("fl_x")
            frame.pop("fl_y")
            frame.pop("cx")
            frame.pop("cy")
            frame.pop("k1")
            frame.pop("k2")
            frame.pop("k3")
            frame.pop("k4")
            frame.pop("h")
            frame.pop("w")

    def add_depth(self, depth_folder=None):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            if depth_folder is None:
                # simply add a path to the depth file modified from the image file path
                depth_file_path = frame["file_path"].replace("images", "depth").replace(".jpg", ".png")
                frame["depth_file_path"] = depth_file_path
            else:
                # add if it exists, otherwise remove
                depth_folder_stem = Path(depth_folder).stem
                depth_file_path = frame["file_path"].replace("images", depth_folder_stem).replace(".jpg", ".png")
                depth_file_path_full = Path(depth_folder).parent / depth_file_path
                if depth_file_path_full.exists():
                    frame["depth_file_path"] = depth_file_path.__str__()
                else:
                    print(f"{depth_file_path_full} not exist")
                    self.traj["frames"].remove(frame)

    def add_normal(self, normal_folder=None):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            if normal_folder is None:
                # simply add a path to the depth file modified from the image file path
                normal_file_path = frame["file_path"].replace("images", "normal").replace(".jpg", ".png")
                frame["normal_file_path"] = normal_file_path
            else:
                # only add if it exists, otherwise remove
                normal_folder_stem = Path(normal_folder).stem
                normal_file_path = frame["file_path"].replace("images", normal_folder_stem).replace(".jpg", ".png")
                normal_file_path_full = Path(normal_folder).parent / normal_file_path
                if normal_file_path_full.exists():
                    frame["normal_file_path"] = normal_file_path.__str__()
                else:
                    print(f"{normal_file_path_full} not exist")
                    self.traj["frames"].remove(frame)

    def add_mask(self):
        for frame in self.traj["frames"]:
            frame["mask_path"] = frame["file_path"].replace("images", "masks")

    def get_clouds_in_json(self, cloud_dir, output_dir):
        cloud_dir = Path(cloud_dir)
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in self.traj["frames"]:
            # find / in file_path
            img_path = Path(frame["file_path"])

            cloud_path = cloud_dir / img_path.parent.name / img_path.name.replace(".jpg", ".pcd")
            output_path = output_dir / img_path.parent.name / img_path.name.replace(".jpg", ".pcd")
            # remove if exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if cloud_path.exists():
                shutil.copy(cloud_path, output_dir / cloud_path.parent.name / cloud_path.name)
            else:
                test_path = cloud_path.parent / (cloud_path.stem[:-1] + cloud_path.suffix)
                if test_path.exists():
                    shutil.copy(test_path, output_dir / cloud_path.parent.name / test_path.name)
                else:
                    print(f"{cloud_path} not exist")

    def get_images_in_json(self, image_dir, output_dir):
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in self.traj["frames"]:
            # find / in file_path
            img_path = image_dir / frame["file_path"]
            assert img_path.exists(), f"{img_path} not exist"

            output_path = output_dir / img_path.parent.name / img_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if img_path.exists():
                shutil.copy(img_path, output_dir / img_path.parent.name / img_path.name)
            else:
                print(f"{img_path} not exist")

    def update_hw(self):
        for frame in self.traj["frames"]:
            frame["h"] = 935
            # frame["w"] = w

    def write_pose_cloud(self, output_file):
        import open3d as o3d

        output_cloud = o3d.geometry.PointCloud()
        for frame in self.traj["frames"]:
            xyz = get_xyz(frame)
            output_cloud.points.append(xyz)
        o3d.io.write_point_cloud(str(output_file), output_cloud)
