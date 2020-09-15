FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV HOME /root

#####################
# PYTORCH FROM SOURCE
#####################
ENV ENV_NAME=base
ENV TORCH_VERSION=v1.6.0-rc1
# FOR SOME REASON TORCH FAILS ("NO CUDA DRIVERS ...) IF CHANGING THE PYTHON VERSION HERE!
# not sure how much the above stuff depends on the same version / whether I have to or can define it further above
#ENV PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN git clone --recursive --branch ${TORCH_VERSION} https://github.com/pytorch/pytorch \
    && cd pytorch \
    && git submodule sync \
    && git submodule update --init --recursive \
    && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
       CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
       conda run -n ${ENV_NAME} python setup.py install \
    && cd .. \
    && rm -rf pytorch/ \
    && conda install -n ${ENV_NAME} cudatoolkit=10.2 -c pytorch \
    && conda clean -afy

#####################
# INSTALL CONDA
#####################
#RUN apt-get update --fix-missing && \
#    apt-get install -y wget && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
#
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh && \
#    /opt/conda/bin/conda clean -tipsy && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
#
#ENV PATH /opt/conda/bin:$PATH

#RUN apt-get update \
#    && apt-get install -y git vim build-essential python-minimal python-pip \
#    && rm -rf /var/lib/apt/lists/*



#####################
# Carla
#####################
# Install dependencies.
RUN apt-get update \
    && apt-get install -y wget software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add - \
    && apt-add-repository "deb http://apt.llvm.org/$(lsb_release -c --short)/ llvm-toolchain-$(lsb_release -c --short)-8 main" \
    && apt-get update

# Additional dependencies for Ubuntu 18.04.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential clang-8 lld-8 g++-7 cmake ninja-build libvulkan1 python python-pip python-dev python3-dev python3-pip libpng-dev libtiff5-dev libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev libxerces-c-dev git vim \
    && pip2 install --user setuptools \
    && pip3 install --user -Iv setuptools==47.3.1 \
    # dependencies for pygame
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

## Additional dependencies for previous Ubuntu versions.
#sudo apt-get install build-essential clang-8 lld-8 g++-7 cmake ninja-build libvulkan1 python python-pip python-dev python3-dev python3-pip libpng16-dev libtiff5-dev libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev libxerces-c-dev &&
#pip2 install --user setuptools &&
#pip3 install --user -Iv setuptools==47.3.1 &&
#pip2 install --user distro &&
#pip3 install --user distro

# Change default clang version.
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180 \
    && update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180

# Get a GitHub and a UE account, and link both.
# Install git.

## Download Unreal Engine 4.24.
#git clone --depth=1 -b 4.24 https://github.com/EpicGames/UnrealEngine.git ~/UnrealEngine_4.24
#cd ~/UnrealEngine_4.24
#
## Download and install the UE patch
#wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/UE_Patch/430667-13636743-patch.txt ~/430667-13636743-patch.txt
#patch --strip=4 < ~/430667-13636743-patch.txt
#
## Build UE
#./Setup.sh && ./GenerateProjectFiles.sh && make
#
## Open the UE Editor to check everything works properly.
#cd ~/UnrealEngine_4.24/Engine/Binaries/Linux && ./UE4Editor

## Clone the CARLA repository.
#git clone https://github.com/carla-simulator/carla

#RUN conda create -n $ENV_NAME python=3.5 \
RUN pip install pygame distro \
#    && pip2 install distro \
#    && pip3 install distro
    && python2 -m pip install distro

RUN git clone https://github.com/carla-simulator/carla.git

#RUN mkdir build
#WORKDIR carla/build
WORKDIR carla
#RUN CPLUS_INCLUDE_PATH="/opt/conda/envs/${ENV_NAME}/include/python3.5m" make setup
RUN CPLUS_INCLUDE_PATH="/opt/conda/include/python3.8m" make setup

#
## Get the CARLA assets.
#cd ~/carla
#./Update.sh
#
## Set the environment variable.
#export UE4_ROOT=~/UnrealEngine_4.24

# make the CARLA server and the CARLA client.
#make launch
RUN make PythonAPI

## Press play in the Editor to initialize the server, and run an example script to test CARLA.
#cd PythonAPI/examples
#python3 spawn_npc.py
RUN easy_install PythonAPI/carla/dist/carla-0.9.10-py3.8-linux-x86_64.egg

#####################
# My Oatomobile
#####################
# other dependencies
#RUN pip install umsgpack wandb tensorboard

WORKDIR /workspace
RUN mkdir oatomobile
WORKDIR /workspace/oatomobile
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV CARLA_ROOT=/carla

COPY . .
#RUN pip install oatomobile
ENV PYTHONPATH=$PYTHONPATH:/workspace/oatomobile

#####################
# CMD
#####################
CMD bash