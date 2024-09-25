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
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-catkin-tools \
    ros-noetic-catkin &&\
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    rosdep init && \
    rosdep update

# create catkin workspace
ARG CATKIN_WS_DIR=/home/catkin_ws
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

# Install fast_lio dependences
RUN git clone https://github.com/ori-drs/livox_ros_driver2.git ${CATKIN_WS_DIR}/src/livox_ros_driver2
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git ${CATKIN_WS_DIR}/src/Livox-SDK2
RUN cd ${CATKIN_WS_DIR}/src/Livox-SDK2/ && mkdir build && cd build && cmake .. && make -j && make install

# Install lio_sam dependences
RUN apt-get update
RUN apt-get install libboost-all-dev
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:borglab/gtsam-release-4.0
RUN apt-get update
RUN apt-get install libgtsam-dev libgtsam-unstable-dev -y

# Install ros packages
RUN git clone https://github.com/catkin/catkin_simple.git ${CATKIN_WS_DIR}/src/catkin_simple 
RUN git clone https://github.com/ori-drs/FAST_LIO.git -b ros1 ${CATKIN_WS_DIR}/src/FAST_LIO
RUN cd ${CATKIN_WS_DIR}/src/FAST_LIO && git submodule update --init
RUN git clone https://github.com/ori-drs/LIO_SAM_6AXIS.git -b dev/frontier_hesai_ho ${CATKIN_WS_DIR}/src/LIO_SAM_6AXIS

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

RUN cd ${CATKIN_WS_DIR} && catkin build livox_ros_driver2
RUN cd ${CATKIN_WS_DIR} && catkin build fast_lio

RUN apt-get update
RUN apt-get install wget
RUN cd ${CATKIN_WS_DIR}/src && wget https://sourceforge.net/projects/geographiclib/files/distrib-C++/GeographicLib-2.4.tar.gz && tar xfpz GeographicLib-2.4.tar.gz
RUN cd ${CATKIN_WS_DIR}/src/GeographicLib-2.4 && mkdir BUILD && cd BUILD && cmake .. && make && make install

RUN cd ${CATKIN_WS_DIR} && catkin build lio_sam_6axis

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
RUN cd ${CATKIN_WS_DIR}/src/FAST_LIO && mkdir config
COPY ./spires_ros/fast_lio2/hesai.yaml ${CATKIN_WS_DIR}/src/FAST_LIO/config
COPY ./spires_ros/lio_sam/launch_lsam ${CATKIN_WS_DIR}/src/LIO_SAM_6AXIS/LIO-SAM-6AXIS/launch
COPY ./spires_ros/lio_sam/config_lsam ${CATKIN_WS_DIR}/src/LIO_SAM_6AXIS/LIO-SAM-6AXIS/config
COPY ./pyproject.toml ${SPIRES_DIR}/pyproject.toml
# RUN pip install -e .

# Make the outputs of the container match the host
RUN chown -R ${UID}:${GID} ${SPIRES_DIR}
USER ${UNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc
RUN echo "alias recon_benchmark=\"python ${SPIRES_DIR}/scripts/reconstruction_benchmark/main.py\"" >> ~/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]