#!/usr/bin/env bash

if (( "$1" == "carla" )); then
  echo "starting carla server"
  xhost +local: && sudo -E docker run --privileged --rm --gpus all -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:latest
elif (( "$1" == "client" )); then
  echo "starting mycarla container"
#  docker run --privileged --rm --gpus all -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw dhonerkamp/mycarla:latest
  cmd='python oatomobile/myscripts/collect_data.py -n 1 --num_steps 100000 --logdir /oatomobile/logs'
  docker run --privileged --rm --gpus all --net=host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --mount type=bind,source="$(pwd)"/logs,target=/oatomobile/logs \
    dhonerkamp/mycarla:latest ${cmd}
else
  echo "Unknown or missing first argument" && exit 1
fi
