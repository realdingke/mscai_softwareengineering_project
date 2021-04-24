#!/bin/bash
app="docker_mscai"
docker build -t ${app} .
nvidia_docker run -it -d -p 5000:5000 \
  --name=${app} \
  -v $PWD:/home/user/src ${app}
