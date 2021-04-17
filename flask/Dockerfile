#
# Copyright (c) 2020 Cord Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# CUDA
# FROM nvidia/cuda:10.1-runtime-ubuntu18.04
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib64
# ENV PATH=${PATH}:/usr/local/nvidia/bin

# Python installation
FROM python:3.7

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev
RUN apt-get install -y python3.7
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libgl1-mesa-glx


COPY ./requirements.txt /app/requirements.txt
# COPY ./configurations /app/configurations
RUN pip3 install --user cython numpy opencv-contrib-python
RUN pip3 install --user pycocotools pynvx
RUN pip3 install --user waitress


# RUN pip3 install --user --no-warn-script-location torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
RUN find /app -iname '*.py' -exec pyminifier '{}' ';'

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
