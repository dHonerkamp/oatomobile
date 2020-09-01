#!/usr/bin/env bash

if (( "$1" == "carla" )); then
  echo "starting carla server"
  xhost +local: && sudo -E docker run --privileged --rm --gpus all -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:latest
elif (( "$1" == "client" )); then
  echo "starting mycarla container"
  docker run --privileged --rm --gpus all -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw dhonerkamp/mycarla:latest
else
  echo "Unknown or missing first argument" && exit 1
fi
