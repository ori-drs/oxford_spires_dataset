FROM ubuntu:22.04 as base

ARG PYTHON_VERSION=3.10
ARG GID
ARG UID
ARG USERNAME
ARG DOCKER_HOME_DIR
ARG DOCKER_WORKDIR

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python-is-python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

FROM base as cpp_build

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
RUN git clone https://github.com/OctoMap/octomap.git --branch v1.10.0 /home/${USERNAME}/octomap && \
    cd /home/${USERNAME}/octomap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    # temporary fix for missing dynamic library
    echo "/usr/local/lib" | tee /etc/ld.so.conf.d/octomap.conf && \
    ldconfig

FROM cpp_build as full_build

WORKDIR ${DOCKER_WORKDIR}
COPY ./oxspires_tools/ ./oxspires_tools/
COPY ./pyproject.toml .
COPY ./CMakeLists.txt ${DOCKER_WORKDIR}/CMakeLists.txt
COPY ./tests ${DOCKER_WORKDIR}/tests
RUN BUILD_CPP=1 pip install -e .

#####################
# Development image #
#####################
FROM full_build as dev

ENV DEBIAN_FRONTEND=dialog

RUN apt-get update && apt-get install -y --no-install-recommends \
RUN apt-get update && apt-get install -y \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install go-task-bin

RUN addgroup --gid $GID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USERNAME

RUN chown -R ${UID}:${GID} ${DOCKER_HOME_DIR}
USER ${USERNAME}

# Colourful bash prompt and ls output
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc \
 && echo "alias ls='ls --color=auto'" >> ~/.bashrc
