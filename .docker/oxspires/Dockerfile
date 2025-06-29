FROM ubuntu:22.04 as base

ARG PYTHON_VERSION=3.10
ARG GID
ARG UID
ARG USERNAME
ARG HOME_DIR
ARG OXSPIRES_DIR

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python-is-python3 \
    vim \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM base as cpp_build
# Dependency for Cpp utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    libeigen3-dev \
    libpcl-dev &&\
    rm -rf /var/lib/apt/lists/*
# Octomap
RUN git clone https://github.com/OctoMap/octomap.git --branch v1.10.0 /home/${USERNAME}/octomap &&\
    cd /home/${USERNAME}/octomap &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make &&\
    make install &&\
    # temporary fix for missing dynamic library
    echo "/usr/local/lib" | tee /etc/ld.so.conf.d/octomap.conf &&\
    ldconfig


FROM cpp_build as final
RUN addgroup --gid $GID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USERNAME

WORKDIR ${OXSPIRES_DIR}

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./oxspires_tools/ ./oxspires_tools/
COPY ./pyproject.toml .
COPY ./CMakeLists.txt ${OXSPIRES_DIR}/CMakeLists.txt
COPY ./tests ${OXSPIRES_DIR}/tests
RUN pip install -e .

RUN chown -R ${UID}:${GID} ${HOME_DIR}/${USERNAME}
USER ${USERNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
