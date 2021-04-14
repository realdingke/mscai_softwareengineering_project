ARG IMAGE_NAME
FROM nvidia/cuda:10.2-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
CMD nvidia-smi

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.2 \
libcublas-dev=10.2.2.89-1 \
&& \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# CT: 4/17


# Install Miniconda
# RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
# MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.11-Linux-x86_64.sh
# MINICONDA_PREFIX=/usr/local
# RUN conda --version
# RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
#  && chmod +x Miniconda3-4.5.11-Linux-x86_64.sh \
#  && Miniconda3-4.5.11-Linux-x86_64.sh -b -f -p $MINICONDA_PREFIX
 
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda --version

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
# RUN conda install conda=4.9.2 \
#  && conda clean -ya
RUN conda list --revision

RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

RUN conda init

RUN pip install cython==0.29.21
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN mkdir -p /home/user/.cache/torch/checkpoints/ \
 && wget http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth -O /home/user/.cache/torch/checkpoints/dla34-ba72cf86.pth

COPY --chown=user:user . /home/user
WORKDIR /home/user/src
# RUN pip install -e git+https://github.com/CharlesShang/DCNv2@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2 --user
CMD ["/bin/bash"]
