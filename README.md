# Oxford Spires Dataset
This repository contains scripts that are used to evaluate localisation, 3D reconstruction and radiance field methods using the [Oxford Spires Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/), accepted in the International Journal of Robotics Research (IJRR).
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
The localisation benchmark runs LiDAR SLAM methods ([Fast-LIO-SLAM](https://github.com/ori-drs/FAST_LIO_SLAM), [SC-LIO-SAM](https://github.com/ori-drs/SC-LIO-SAM), [ImMesh](https://github.com/ori-drs/ImMesh_hesai)), a LIVO method ([Fast-LIVO2](https://github.com/ori-drs/FAST-LIVO2)) and LiDAR Bundle Adjustment method ([HBA](https://github.com/ori-drs/HBA)). The resultant trajectory are evaluated against the ground truth trajectory using [evo](https://github.com/MichaelGrupp/evo).

Each link provided for the methods above is a fork containing a branch `config-used-OSD` with the configurations used for the evaluation.

Build the docker container and run the methods:
```bash
git clone https://github.com/ori-drs/oxford_spires_dataset.git
cd oxford_spires_dataset
docker compose -f .docker/loc/docker-compose.yml run --build oxspires_loc

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
docker compose -f .docker/recon/docker-compose.yml run --build oxspires_recon

# inside the docker
python scripts/reconstruction_benchmark/main.py --config-file config/recon_benchmark.yaml
```

## Novel-view Synthesis Benchmark
```bash
# This will download data from Hugging Face first
python scripts/reconstruction_benchmark/nvs_benchmark.py
```
the NVS benchmakr is also included in the reconstruction benchmark script, since it builds upon output from COLMAP. 

## Just Use oxspires_tools  
To just install `oxspires_tools`, build the docker container:
```bash
docker compose -f .docker/oxspires/docker-compose.yml run --build oxspires_utils
```

Installing `oxspires_tools` requires PCL and Octomap installation. If you just want to install the python package, run:
```bash
BUILD_CPP=0 pip install .
```

## Contributing
Please refer to the [contributing](docs/contributing.md) page.

## Citation
```bibtex
@article{tao2025spires,
title={The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods},
author={Tao, Yifu and Mu{\~n}oz-Ba{\~n}{\'o}n, Miguel {\'A}ngel and Zhang, Lintong and Wang, Jiahao and Fu, Lanke Frank Tarimo and Fallon, Maurice},
journal={International Journal of Robotics Research}, 
year={2025},
}
```