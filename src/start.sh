#!/bin/bash
app="docker_mscai"
cd ..
docker build -t ${app} .
nvidia-docker run -it -d -p 5000:5000 \
  --name=${app} \
  -v $PWD/src:/home/user/src ${app}
docker exec -it ${app} python flasktest.py
