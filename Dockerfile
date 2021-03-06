# ARG IMAGE_NAME
# # FROM nvidia/cuda:10.2-runtime-ubuntu18.04
# # FROM nvidia/cuda:11.0-runtime-ubuntu18.04
# # LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
# # CMD nvidia-smi # not in fairmot
# ARG UBUNTU_VERSION=18.04
# ARG ARCH=
# ARG CUDA=11.0
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARG ARCH
# ARG CUDA
# # ARG CUDNN=8.0.4.30-1
# ARG CUDNN=8.0.5.39
# ARG CUDNN_MAJOR_VERSION=8
# SHELL ["/bin/bash", "-c"]

# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #         cuda-nvml-dev-$CUDA_PKG_VERSION \
# #         cuda-command-line-tools-$CUDA_PKG_VERSION \
# # cuda-libraries-dev-$CUDA_PKG_VERSION \
# #         cuda-minimal-build-$CUDA_PKG_VERSION \
# #         libnccl-dev=$NCCL_VERSION-1+cuda11.0 \
# # libcublas-dev=10.2.2.89-1 \
# # && \
# #     rm -rf /var/lib/apt/lists/*


# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #         cuda-command-line-tools-${CUDA/./-} \
# #         libcublas-${CUDA/./-} \
# #         cuda-nvrtc-${CUDA/./-} \
# #         libcufft-${CUDA/./-} \
# #         libcurand-${CUDA/./-} \
# #         libcusolver-${CUDA/./-} \
# #         libcusparse-${CUDA/./-} \
# #         libcudnn8=${CUDNN}+cuda${CUDA} \
# #         vim \
# # && \
# #     rm -rf /var/lib/apt/lists/*

# RUN apt update && apt install -y --no-install-recommends \
#     cuda-cudart-11-0 \
#     cuda-nvrtc-11-0 \
#     libcublas-11-0 \
#     libcufft-11-0 \
#     libcurand-11-0 \
#     libcusolver-11-0 \
#     libcusparse-11-0 \
#     cuda-compat-11-0 \
#     cuda-nvtx-11-0 \
#     libgomp1 \
#     && ln -s cuda-11.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* .cache/ && \
#     rm /usr/local/cuda/targets/x86_64-linux/lib/libcusolverMg.so*


# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# # Install some basic utilities
# RUN apt-get update && apt-get install -y \
#     curl \
#     wget \
#     build-essential \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 ffmpeg libsm6 libxext6 \
#  && rm -rf /var/lib/apt/lists/*

# # Create a working directory
# RUN mkdir /app
# WORKDIR /app

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#  && chown -R user:user /app
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

# USER user

# # All users can use /home/user as their home directory
# ENV HOME=/home/user
# RUN chmod 777 /home/user

# # CT: 4/17


# # Install Miniconda
# # RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \

# RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh \
#  && chmod +x ~/miniconda.sh \
#  && ~/miniconda.sh -b -p ~/miniconda \
#  && rm ~/miniconda.sh
# ENV PATH=/home/user/miniconda/bin:$PATH
# ENV CONDA_AUTO_UPDATE_CONDA=false


# # Create a Python 3.6 environment
# RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
#  && /home/user/miniconda/bin/conda clean -ya
 
# # RUN sudo ln -s /home/user/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
# # RUN source activate py36
# #  && /home/user/miniconda/bin/conda clean -ya
# ENV CONDA_DEFAULT_ENV=py36
# ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
# ENV PATH=$CONDA_PREFIX/bin:$PATH
# RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
#  && /home/user/miniconda/bin/conda clean -ya

# # CUDA 10.0-specific steps
# # RUN conda install conda=4.9.2 \
# #  && conda clean -ya
# # RUN conda list --revision

# RUN conda install -y -c pytorch \
#     cudatoolkit=10.0 \
#     "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
#     "torchvision=0.4.0=py36_cu100" \
#  && conda clean -ya

# # Install HDF5 Python bindings
# RUN conda install -y h5py=2.8.0 \
#  && conda clean -ya
# RUN pip install h5py-cache==1.0

# # Install Torchnet, a high-level framework for PyTorch
# RUN pip install torchnet==0.0.4

# # Install Requests, a Python library for making HTTP requests
# RUN conda install -y requests=2.19.1 \
#  && conda clean -ya

# # Install Graphviz
# RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
#  && conda clean -ya

# # Install OpenCV3 Python bindings
# RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#     libgtk2.0-0 \
#     libcanberra-gtk-module \
#  && sudo rm -rf /var/lib/apt/lists/*
# RUN conda install -y -c menpo opencv3=3.1.0 \
#  && conda clean -ya

# RUN conda init

# RUN pip install cython==0.29.21
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# # COPY . .
# # WORKDIR /app/DCNv2
# # RUN ./make.sh

# RUN git clone --recursive https://github.com/CharlesShang/DCNv2
# RUN cd DCNv2 && bash ./make.sh

# RUN mkdir -p /home/user/.cache/torch/checkpoints/ \
#  && wget http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth -O /home/user/.cache/torch/checkpoints/dla34-ba72cf86.pth

# COPY --chown=user:user . /home/user
# WORKDIR /home/user/src

# # RUN pip install -e git+https://github.com/CharlesShang/DCNv2@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2 --user
# CMD ["/bin/bash"]

ARG IMAGE_NAME
FROM nvidia/cuda:10.2-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
# CMD nvidia-smi # not in fairmot

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
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false


# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
 
# RUN sudo ln -s /home/user/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
# RUN source activate py36
#  && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.0-specific steps
# RUN conda install conda=4.9.2 \
#  && conda clean -ya
# RUN conda list --revision

RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

RUN pip install --user waitress

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

# COPY . .
# WORKDIR /app/DCNv2
# RUN ./make.sh

RUN git clone --recursive https://github.com/CharlesShang/DCNv2
RUN cd DCNv2 && bash ./make.sh

RUN mkdir -p /home/user/.cache/torch/checkpoints/ \
 && wget http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth -O /home/user/.cache/torch/checkpoints/dla34-ba72cf86.pth

COPY --chown=user:user . /home/user
WORKDIR /home/user/src

# RUN pip install -e git+https://github.com/CharlesShang/DCNv2@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2 --user
CMD ["/bin/bash"]
