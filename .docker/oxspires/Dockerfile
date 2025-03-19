FROM ubuntu:22.04

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


ARG OXSPIRES_DIR=/home/oxford_spires_dataset
WORKDIR ${OXSPIRES_DIR}

COPY ./scripts/dataset_manager/gui.py .
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./oxford_spires_utils/ ./oxford_spires_utils/
COPY ./pyproject.toml .
RUN pip install .

