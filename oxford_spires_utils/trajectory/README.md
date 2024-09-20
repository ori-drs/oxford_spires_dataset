## Introduction
This package provides utilities for loading and saving different trajectory format, and convert the pose convention from one to the other. Unittest is also provided to validate the results.

## Quick Start
```bash
cd <nerfstudio_drs>/drs_scripts/pose_utils
# configure traj_config.yaml
python traj_handler.py
```

## Pose Convention
1. **Robotics**: 
   - `x forward, y left, z up`
2. **Computer Vision**: 
   - `x right, y down, z foward`
   - Used in COLMAP
3. **Computer Graphics**: 
   - `x right, y up, z backward`
   - Used in NeRF and Blender


## File Format
We add the pose convention as file suffix. e.g. `carla_pose_robotics.txt` to indicate `Robotics` convention
1. **[TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)** txt
   - `timestamp tx ty tz qx qy qz qw`
2. **NeRF** json
   ```    json
   "frames": [
        {
            "file_path": "xxx.jpg",
            "transform_matrix": [[],[],[],[]]
        }
   ```

## Config
Example yaml config
```       yaml
input_traj:
  file_format: "tum"
  file_path: "<nerfstudio_drs>/drs_scripts/pose_utils/test/input_traj/carla_pose_nerf.txt"
  pose_convention: "nerf"
  visualise: true
  optional:
    tum_reader_type: "custom"
    tum_custom_reader_prefix: ""
    tum_custom_reader_suffix: "_front"

output_traj:
  file_format: "tum"
  file_path: "/home/yifu/workspace/nerfstudio_drs/drs_scripts/pose_utils/test/output_traj/carla_pose_robotics.txt"
  pose_convention: "robotics"
  visualise: false
```

- `file_format`: ["tum", "nerf"]

- `file_path`: str
  -  path to trajectory file

- `pose_convention`: ["robotics", "nerf", "colmap"]

- `visualise`: bool
  -  whether visualise trajectory in open3d

- `optional`

  -  `tum_reader_type`: ["custom","evo"]
     -  "custom" use a custom tum handler that supports prefix/suffix for timestamp
     -  "evo": use evo's tum loader

  -  `tum_custom_reader_prefix`: str
     -  timestamp prefix for custom tum reader

  -  `tum_custom_reader_suffix`: str
     -  timestamp prefix for custom tum reader
  - `nerf_writer_template_output_path`: str
    - the template for output file with the parameters like intrinsics. Usually the input nerf trajectory. Nothing is changed apart from the `transform_matrix` under "frames". Note that this does not support multi-camera settings yet due to timestamp conflict.
  
## UnitTest
Unittest is provided for the trajectory readers and writers. 

```bash
cd <nerfstudio_drs>/drs_scripts/pose_utils
python test.py
```

Tests implemented:
1. read tum pose, try identity transform and writes as tum pose; check if the pose is the same
2. read tum pose, transform to a new pose convention and writes as tum pose, read the new pose and convert back to the old pose convention; check if the pose is the same
3. read nerf pose, try identity transform and write as nerf pose; check if the pose is the same
4. read nerf pose, transform to a new pose convention and writes as nerf pose, read the new pose and convert back to the old pose convention; check if the pose is the same