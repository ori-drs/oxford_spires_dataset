FROM ubuntu:22.04 as base

ARG PYTHON_VERSION=3.10

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python-is-python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

FROM base as cpp_build

ARG DOCKER_USERNAME
ARG DOCKER_HOME_DIR

# Dependency for Cpp utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libeigen3-dev \
    libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

# Octomap
ARG OCTOMAP_DIR=${DOCKER_HOME_DIR}/octomap
RUN git clone https://github.com/OctoMap/octomap.git --branch v1.10.0 ${OCTOMAP_DIR} && \
    cd ${OCTOMAP_DIR} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    # temporary fix for missing dynamic library
    echo "/usr/local/lib" | tee /etc/ld.so.conf.d/octomap.conf && \
    ldconfig

FROM cpp_build as full_build

ARG DOCKER_USERNAME
ARG DOCKER_HOME_DIR

WORKDIR ${DOCKER_HOME_DIR}/oxford_spires_dataset
COPY ./oxspires_tools/ ./oxspires_tools/
COPY ./pyproject.toml .
COPY ./CMakeLists.txt .
COPY ./tests ./tests
RUN BUILD_CPP=1 pip install -e .

#####################
# Development image #
#####################
FROM full_build as dev

ARG DOCKER_GID
ARG DOCKER_UID
ARG DOCKER_USERNAME
ARG DOCKER_HOME_DIR

ENV DEBIAN_FRONTEND=dialog

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $DOCKER_GID $DOCKER_USERNAME \
 && adduser --disabled-password --gecos '' --uid $DOCKER_UID --gid $DOCKER_GID $DOCKER_USERNAME \
 && echo $DOCKER_USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DOCKER_USERNAME

RUN pip install --ignore-installed -e .[dev]

RUN chown -R ${DOCKER_UID}:${DOCKER_GID} ${DOCKER_HOME_DIR}
USER ${DOCKER_USERNAME}

# Colourful bash prompt and ls output
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc \
 && echo "alias ls='ls --color=auto'" >> ~/.bashrc
