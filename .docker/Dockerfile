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


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libpcl-dev &&\
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/OctoMap/octomap.git --branch v1.10.0 ${HOME_DIR}/octomap &&\
    cd ${HOME_DIR}/octomap &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make &&\
    make install

####################
# Deployment image #
####################
FROM base as deploy

ARG GID
ARG UID
ENV UNAME=docker_dev
RUN addgroup --gid $GID $UNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME

ARG SPIRES_DIR=/home/docker_dev/oxford_spires_dataset
WORKDIR ${SPIRES_DIR}

COPY ./requirements.txt ${SPIRES_DIR}/requirements.txt
RUN pip install -r requirements.txt
COPY ./oxford_spires_utils/ ${SPIRES_DIR}/oxford_spires_utils/
COPY ./spires_cpp ${SPIRES_DIR}/spires_cpp/
COPY ./pyproject.toml ${SPIRES_DIR}/pyproject.toml
RUN cd ${SPIRES_DIR}/spires_cpp && pip install -e . 
RUN pip install -e .

# Make the outputs of the container match the host

RUN chown -R ${UID}:${GID} ${SPIRES_DIR}/*
USER ${UNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc

CMD ["/bin/bash"]