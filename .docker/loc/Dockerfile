##############
# Base image #
##############
ARG UBUNTU_VERSION=20.04
ARG NVIDIA_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG GID
ARG UID
ARG USERNAME
ARG DOCKER_HOME_DIR
ARG DOCKER_WORKDIR
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
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-catkin-tools \
    ros-noetic-catkin &&\
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    rosdep init && \
    rosdep update

# create catkin workspace
ARG CATKIN_WS_DIR=${DOCKER_HOME_DIR}/catkin_ws
RUN . /opt/ros/noetic/setup.sh && mkdir -p ${CATKIN_WS_DIR}/src && cd ${CATKIN_WS_DIR} && catkin build && \
    echo "source ${CATKIN_WS_DIR}/devel/setup.bash" >> ~/.bashrc

# Install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    python-is-python3 \
    python3-pip \
    libyaml-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install C++ Dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-iostreams-dev \
    libboost-numpy-dev \
    libboost-python-dev \
    libboost-system-dev \
    libeigen3-dev \
    libtbb-dev \
    libgflags-dev \
    libgl1-mesa-glx \
    libgoogle-glog-dev \
    protobuf-compiler \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

######################
# Install SC-LIO-SAM #
######################
# Install lio_sam dependences
RUN apt-get update
RUN apt-get install libboost-all-dev
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:borglab/gtsam-release-4.0
RUN apt-get update
RUN apt-get install libgtsam-dev libgtsam-unstable-dev -y
RUN apt-get update
RUN apt-get install wget
RUN cd ${CATKIN_WS_DIR}/src && wget https://sourceforge.net/projects/geographiclib/files/distrib-C++/GeographicLib-2.4.tar.gz && tar xfpz GeographicLib-2.4.tar.gz
RUN cd ${CATKIN_WS_DIR}/src/GeographicLib-2.4 && mkdir BUILD && cd BUILD && cmake .. && make && make install
# Install ros packages
RUN git clone https://github.com/ori-drs/SC-LIO-SAM.git ${CATKIN_WS_DIR}/src/SC-LIO-SAM
RUN cd ${CATKIN_WS_DIR} && catkin build lio_sam

#########################
# Install Fast-LIO-SLAM #
#########################
# Install fast_lio dependences
RUN git clone https://github.com/Livox-SDK/livox_ros_driver ${CATKIN_WS_DIR}/src/livox_ros_driver
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git ${CATKIN_WS_DIR}/src/Livox-SDK2
RUN cd ${CATKIN_WS_DIR}/src/Livox-SDK2/ && mkdir build && cd build && cmake .. && make -j && make install
RUN git clone https://ceres-solver.googlesource.com/ceres-solver -b 2.0.0 ${CATKIN_WS_DIR}/src/ceres-solver
RUN apt-get install libgoogle-glog-dev libgflags-dev
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libeigen3-dev
RUN cd ${CATKIN_WS_DIR}/src/ceres-solver/ && mkdir build && cd build && cmake .. && make -j3 && make install
# Install ros packages
RUN git clone https://github.com/ori-drs/FAST_LIO_SLAM.git ${CATKIN_WS_DIR}/src/FAST_LIO_SLAM
RUN cd ${CATKIN_WS_DIR}/src/FAST_LIO_SLAM && git submodule update --init
RUN cd ${CATKIN_WS_DIR} && catkin build livox_ros_driver
RUN cd ${CATKIN_WS_DIR} && catkin build fast_lio
RUN cd ${CATKIN_WS_DIR} && catkin build aloam_velodyne

###############
# Install HBA #
###############
# Install ros packages
RUN git clone https://github.com/ori-drs/HBA.git ${CATKIN_WS_DIR}/src/HBA
RUN cd ${CATKIN_WS_DIR} && catkin build hba

# Trajectory evaluation dependences
RUN apt-get update
RUN apt-get install tmux -y
RUN pip install libtmux
RUN pip install evo

# Final setup
RUN addgroup --gid $GID $USERNAME
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USERNAME

WORKDIR ${DOCKER_WORKDIR}

COPY ./oxspires_tools/ ${DOCKER_WORKDIR}/oxspires_tools/
COPY ./pyproject.toml ${DOCKER_WORKDIR}/pyproject.toml
# RUN pip install -e .

# Make the outputs of the container match the host
RUN chown -R ${UID}:${GID} ${DOCKER_HOME_DIR}
USER ${USERNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "alias recon_benchmark=\"python ${DOCKER_WORKDIR}/scripts/reconstruction_benchmark/main.py\"" >> ~/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]