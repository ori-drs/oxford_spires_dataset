# Oxford Spires Dataset
This repository contains scripts that are used to evaluate localisation, 3D reconstruction and radiance field methods using the [Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/).
- [Website](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires)
- [Arxiv](https://arxiv.org/abs/2411.10546)
- [Video](https://www.youtube.com/watch?v=AKZ-YrOob_4)

This is a pre-release of the software. The codebase will be refactored in the near future. Please feel free to ask questions about the dataset and report bugs in the Github Issues.

## Download
You can download the dataset from [HuggingFace](https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/tree/main) with this script. Define which folder to download by changing the `example_pattern`.
```bash
python scripts/dataset_download.py
```
You can also download the dataset from [Google Drive](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/#googledrive).

## Localisation Benchmark
The localisation benchmark runs LiDAR SLAM methods ([Fast-LIO-SLAM](https://github.com/gisbi-kim/FAST_LIO_SLAM), [SC-LIO-SAM](https://github.com/gisbi-kim/SC-LIO-SAM), [ImMesh](https://github.com/ori-drs/ImMesh_hesai)) and LiDAR Bundle Adjustment method ([HBA](https://github.com/hku-mars/HBA)). The resultant trajectory are evaluated against the ground truth trajectory using [evo](https://github.com/MichaelGrupp/evo).

Build the docker container and run the methods:
```bash
cd oxford_spires_dataset
docker compose -f .docker_loc/docker-compose.yml run --build spires

# in the docker
python scripts/localisation_benchmark/colmap.py
python scripts/localisation_benchmark/fast_lio_slam.py
python scripts/localisation_benchmark/immesh.py
python scripts/localisation_benchmark/vilens_hba.py
```


## Reconstruction Benchmark
The reconstruction benchmark runs Structure-from-Motion ([COLMAP](https://colmap.github.io/)), Multi-view Stereo ([OpenMVS](https://github.com/cdcseacave/openMVS)), radiance field methods ([Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/main/nerfstudio)'s Nerfacto and Splatfacto), and generates 3D point cloud reconstruction, which is evaluated against the TLS-captured ground truth 3D point cloud.

Build the docker container and run the methods:
```bash
cd oxford_spires_dataset
docker compose -f .docker/docker-compose.yml run --build spires

# inside the docker
python scripts/reconstruction_benchmark/main.py --config-file config/recon_benchmark.yaml
```

## Novel-view Synthesis Benchmark
Currently, the NVS benchmakr is included in the reconstruction benchmark script, since it builds upon output from COLMAP. 

## Contributing
Please refer to [Angular's guide](https://github.com/angular/angular/blob/22b96b96902e1a42ee8c5e807720424abad3082a/CONTRIBUTING.md) for contributing.

### Formatting
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter for Python, and Clang for C++.

Installing [`pre-commit`](https://pre-commit.com/) will format and lint your code before you commit:
```bash
# pip install pre-commit
pre-commit install
```
Alternatively, you can also run Ruff manually:
```bash
# pip install ruff
ruff check --fix --config pyproject.toml
ruff format --config pyproject.toml
```
