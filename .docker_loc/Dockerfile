##############
# Base image #
##############
ARG UBUNTU_VERSION=20.04
ARG NVIDIA_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Install ROS
ARG ROS_DISTRO=noetic
# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    lsb-release \
    curl && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt install -y \
    ros-noetic-ros-base \
    python3-rosdep \
    python3-catkin-tools \
    ros-noetic-catkin &&\
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    rosdep init && \
    rosdep update

# Install tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python-is-python3 \
    python3.8-dev \
    python3-pip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11

# create catkin workspace
ARG CATKIN_WS_DIR=/home/catkin_ws
RUN . /opt/ros/noetic/setup.sh && mkdir -p ${CATKIN_WS_DIR}/src && cd ${CATKIN_WS_DIR} && catkin build && \
    echo "source ${CATKIN_WS_DIR}/devel/setup.bash" >> ~/.bashrc
# create requered packages
RUN git clone https://github.com/ori-drs/FAST_LIO.git ${CATKIN_WS_DIR}/src/FAST_LIO
RUN cd ${CATKIN_WS_DIR} && catkin build

# Trajectory evaluation dependences
RUN apt-get update
RUN apt-get install tmux -y
RUN pip install libtmux
RUN pip install evo

# Final setup
ARG GID
ARG UID
ENV UNAME=docker_dev
RUN addgroup --gid $GID $UNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $UNAME

ARG SPIRES_DIR=/home/oxford_spires_dataset
WORKDIR ${SPIRES_DIR}

COPY ./oxford_spires_utils/ ${SPIRES_DIR}/oxford_spires_utils/
COPY ./spires_ros/fast_lio2/launch_osd ${CATKIN_WS_DIR}/src/FAST_LIO/launch
COPY ./spires_ros/fast_lio2/hesai.yaml ${CATKIN_WS_DIR}/src/FAST_LIO/config
COPY ./pyproject.toml ${SPIRES_DIR}/pyproject.toml
RUN pip install -e .

# Make the outputs of the container match the host

RUN chown -R ${UID}:${GID} ${SPIRES_DIR}
USER ${UNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "alias recon_benchmark=\"python ${SPIRES_DIR}/scripts/reconstruction_benchmark/main.py\"" >> ~/.bashrc

CMD ["/bin/bash"]