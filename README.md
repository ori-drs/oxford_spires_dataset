# Oxford Spires Dataset
This repository contains scripts that are used to evaluate Lidar/Visual SLAM on the Oxford Spires Dataset.

## Installation
### oxford_spires_utils (Python)
```bash
pip install -e .
```

### octomap_utils (C++)
Install octomap to your system
```bash
git clone https://github.com/OctoMap/octomap.git && cd octomap
mkdir build && cd build
cmake ..
make
sudo make install
```
Then install the sripts in Spires.
```bash
cd <oxford_spires_dataset>/octomap_utils
mkdir build && cd build
cmake ..
make
```

## Usage
### octomap_utils
```bash
cd <oxford_spires_dataset>/octomap_utils/build
./pcd2bt <path_to_pcd_folder> -r [resolution] -s [saved_path] -tf [x,y,z, quat_w, quat_x, quat_y, quat_z]
```

```bash
cd <oxford_spires_dataset>/octomap_utils/build
./octomap_utils/get_occ_free_from_bt.cpp <path_to_bt_file> -sf <path_to_free_pcd> -so <path_to_occupied_pcd>
```

## Contributing
Please refer to Angular's guide for contributing(https://github.com/angular/angular/blob/22b96b96902e1a42ee8c5e807720424abad3082a/CONTRIBUTING.md).

### Formatting
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter. Installing the `pre-commit` will format and lint your code before you commit:

```bash
$ pip install pre-commit
$ pre-commit install
```