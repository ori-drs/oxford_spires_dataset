FROM ubuntu:22.04 as base

# Set Python version as build argument with default
ARG PYTHON_VERSION=3.10

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

ARG GID
ARG UID
ARG USERNAME
RUN addgroup --gid $GID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USERNAME
USER ${USERNAME}

ARG OXSPIRES_DIR=/home/${USERNAME}/oxford_spires_dataset
WORKDIR ${OXSPIRES_DIR}

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./oxford_spires_utils/ ./oxford_spires_utils/
COPY ./pyproject.toml .
RUN pip install .

RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
