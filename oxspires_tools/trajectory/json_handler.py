import json
import shutil
from pathlib import Path

import numpy as np

from oxspires_tools.trajectory.json_handler_submap_utils import get_xyz, save_submap_cluster, viz_submap_cluster


class JsonHandler:
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

    def filter_folder(self, folder_path):
        frames_copy = self.traj["frames"].copy()
        for frame in frames_copy:
            file_path = Path(frame["file_path"])
            if file_path.parent != Path(folder_path):
                self.traj["frames"].remove(frame)
                print(f"removed {file_path} from json")
        print(f"filter_folder {len(frames_copy)} files in json, {len(self.traj['frames'])} left")

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

    def split_into_submaps(self, save_dir, cluster_method="spectral", spectral_n_clusters=5, overlap_dist=0):
        xyz_list = np.array([get_xyz(frame) for frame in self.traj["frames"]])
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if cluster_method == "naive":
            submap_radius = 40
            submap_node_indices = [0]
            submap_node_xyz = [get_xyz(self.traj["frames"][0])]
            for i, frame in enumerate(self.traj["frames"]):
                if np.all(np.linalg.norm(get_xyz(frame) - submap_node_xyz, axis=1) > submap_radius):
                    submap_node_indices.append(i)
                    submap_node_xyz.append(get_xyz(frame))
            for i, submap_node_index in enumerate(submap_node_indices):
                output_json = self.traj.copy()
                assert (xyz_list[submap_node_index] == submap_node_xyz[i]).all()
                submap_poses_indices = np.where(np.linalg.norm(xyz_list - submap_node_xyz[i], axis=1) < submap_radius)[
                    0
                ]
                submap_poses = [self.traj["frames"][j] for j in submap_poses_indices]
                output_json["frames"] = submap_poses

                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / f"submap_{i}.json", "w") as f:
                    json.dump(output_json, f, indent=4)

        elif cluster_method == "mean_shift":
            from sklearn.cluster import MeanShift

            ms = MeanShift(bandwidth=19)
            clusters = ms.fit_predict(xyz_list)
            viz_submap_cluster(clusters, xyz_list, save_path=(save_dir / "submap_plt_spectral.png"))
            save_submap_cluster(self.traj, clusters, save_dir, xyz_list)
        elif cluster_method == "spectral":
            from sklearn.cluster import SpectralClustering

            spectral = SpectralClustering(n_clusters=spectral_n_clusters, affinity="nearest_neighbors", random_state=42)
            # Perform clustering
            clusters = spectral.fit_predict(xyz_list)

            viz_submap_cluster(clusters, xyz_list, save_path=(save_dir / "submap_plt_spectral.png"))
            save_submap_cluster(self.traj, clusters, save_dir, xyz_list, overlap_dist=overlap_dist)

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
