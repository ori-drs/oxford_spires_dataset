# Oxford Spires Dataset
This repository contains scripts that are used to evaluate localisation, 3D reconstruction and radiance field methods using the [Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/). 

This is a pre-release of the software. The codebase will be refactored in the near future.

## Localisation Benchmark
The localisation benchmark runs LiDAR SLAM methods ([Fast-LIO-SLAM](https://github.com/gisbi-kim/FAST_LIO_SLAM), [SC-LIO-SAM](https://github.com/gisbi-kim/SC-LIO-SAM)) and LiDAR Bundle Adjustment method ([HBA](https://github.com/hku-mars/HBA)). The resultant trajectory are evaluated against the ground truth trajectory using [evo](https://github.com/MichaelGrupp/evo).


## Reconstruction Benchmark
The reconstruction benchmark runs Structure-from-Motion ([COLMAP](https://colmap.github.io/)), Multi-view Stereo ([OpenMVS](https://github.com/cdcseacave/openMVS)), radiance field methods ([Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/main/nerfstudio)'s Nerfacto and Splatfacto), and generates 3D point cloud reconstruction, which is evaluated against the TLS-captured ground truth 3D point cloud.

Build the docker container and run 
```bash
cd oxford_spires_dataset
docker compose -f .docker/docker-compose.yml run --build spires

# inside the docker
python scripts/reconstruction_benchmark/main.py --config-file config/recon_benchmark.yaml
```

## Novel-view Synthesis Benchmark
Currently, the NVS benchmakr is included in the reconstruction benchmark script, since it builds upon output from COLMAP. 

## Contributing
Please refer to Angular's guide for contributing(https://github.com/angular/angular/blob/22b96b96902e1a42ee8c5e807720424abad3082a/CONTRIBUTING.md).

### Formatting
We use [Ruff](https://github.com/astral-sh/ruff) as the formatter and linter for Python, and Clang for C++. Installing the `pre-commit` will format and lint your code before you commit:

```bash
$ pip install pre-commit
$ pre-commit install
```