ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0

##############
# Base image #
##############
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

ARG ROS_DISTRO=noetic
ARG COLMAP_VERSION=3.8
ARG CUDA_ARCHITECTURES=native
ARG HOME_DIR=/home

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11

##############
# spires_cpp #
##############
from base as spires_cpp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libpcl-dev &&\
    rm -rf /var/lib/apt/lists/*

# Octomap
RUN git clone https://github.com/OctoMap/octomap.git --branch v1.10.0 ${HOME_DIR}/octomap &&\
    cd ${HOME_DIR}/octomap &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make &&\
    make install

##############
#   OpenMVS  #
##############
from spires_cpp as openmvs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Boost
    libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev \
    # OpenCV
    libopencv-dev \
    # CGAL
    libcgal-dev libcgal-qt5-dev &&\
    rm -rf /var/lib/apt/lists/*

ARG OPENMVS_BUILD_ARG="-DOpenMVS_USE_CUDA=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include/ -DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64 -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/"
RUN git clone https://github.com/cdcseacave/VCG.git  ${HOME_DIR}/vcglib &&\
    git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git --branch v2.3.0 ${HOME_DIR}/openMVS  &&\
    cd ${HOME_DIR}/openMVS &&\
    mkdir make &&\
    cd make &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=${HOME_DIR}/vcglib $OPENMVS_BUILD_ARG &&\
    make &&\
    make install

##############
#   COLMAP   #
##############
from openmvs as colmap
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev &&\
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-10 g++-10
RUN export CC=/usr/bin/gcc-10 &&\
    export CXX=/usr/bin/g++-10 &&\
    export CUDAHOSTCXX=/usr/bin/g++-10

# TODO: You might have to change this to your specific GPU architecture. Ideally should not need to specify CUDA_ARCHITECTURES. 
ARG CUDA_ARCHITECTURES=89
RUN git clone https://github.com/colmap/colmap.git --branch 3.8 ${HOME_DIR}/colmap &&\
    cd ${HOME_DIR}/colmap &&\
    mkdir build &&\
    cd build &&\
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} &&\
    ninja &&\
    ninja install

####################
#    nerfstudio    #
####################

FROM colmap as nerfstudio
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 'numpy<2.0.0' --extra-index-url https://download.pytorch.org/whl/cu118
RUN TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" pip install --no-cache-dir "git+https://github.com/NVlabs/tiny-cuda-nn.git@b3473c81396fe927293bdfd5a6be32df8769927c#subdirectory=bindings/torch"

####################
# Deployment image #
####################
FROM nerfstudio as deploy

ARG GID
ARG UID
ENV UNAME=docker_dev
RUN addgroup --gid $GID $UNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME

ARG SPIRES_DIR=/home/oxford_spires_dataset
WORKDIR ${SPIRES_DIR}

COPY ./requirements.txt ${SPIRES_DIR}/requirements.txt
RUN pip install -r requirements.txt
COPY ./oxford_spires_utils/ ${SPIRES_DIR}/oxford_spires_utils/
COPY ./spires_cpp ${SPIRES_DIR}/spires_cpp/
COPY ./pyproject.toml ${SPIRES_DIR}/pyproject.toml
RUN cd ${SPIRES_DIR}/spires_cpp && pip install -e . 
RUN pip install -e .

# Make the outputs of the container match the host

RUN chown -R ${UID}:${GID} ${SPIRES_DIR}
USER ${UNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "alias recon_benchmark=\"python ${SPIRES_DIR}/scripts/reconstruction_benchmark/main.py\"" >> ~/.bashrc

CMD ["/bin/bash"]