# Oxford Spires Dataset
This repository contains scripts that are used to evaluate Lidar/Visual SLAM on the Oxford Spires Dataset.

## Installation
### oxford_spires_utils (Python)
```bash
pip install -e .
```

### spires_cpp (C++ Pybind)
Install [octomap](https://github.com/OctoMap/octomap) and [PCL](https://github.com/PointCloudLibrary/pcl) to your system, then
```bash
cd spires_cpp
pip install -e .
```


## Contributing
Please refer to Angular's guide for contributing(https://github.com/angular/angular/blob/22b96b96902e1a42ee8c5e807720424abad3082a/CONTRIBUTING.md).

### Formatting
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter for Python, and Clang for C++. Installing the `pre-commit` will format and lint your code before you commit:

```bash
$ pip install pre-commit
$ pre-commit install
```