#!/usr/bin/env bash

if (( "$1" == "carla" )); then
  echo "starting carla server"
  xhost +local: && sudo -E docker run --privileged --rm --gpus all -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:latest
elif (( "$1" == "client" )); then
  echo "starting mycarla container"

  towns=('Town01' 'Town02' 'Town03' 'Town04' 'Town05' 'Town06' 'Town07' 'Town10')
  weather=('ClearNoon' 'MidRainyNoon', 'CloudySunset')
  num_steps=10000
  export CARLA_ROOT=/home/honerkam/repos/carla
  export PYTHONPATH=/home/honerkam/repos/oatomobile:$PYTHONPATH
#  towns=('Town05' 'Town06' 'Town07' 'Town10')
  for t in "${towns[@]}"; do
    for w in "${weather[@]}"; do
      cmd="python oatomobile/myscripts/collect_data.py -n 1 --num_steps ${num_steps} --logdir ./logs --town ${t} --occ busyV0 --weather ${w}"
      echo "Running cmd: ${cmd}"
      eval $cmd

  #    cmd="python oatomobile/myscripts/collect_data.py -n 1 --num_steps ${num_steps} --logdir /oatomobile/logs --town ${t} --occ busyV0 --weather ${w}"
  #    echo "Running cmd: ${cmd}"
  #    docker run --privileged --rm --gpus all --net=host -e DISPLAY=$DISPLAY \
  #      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  #      --mount type=bind,source="$(pwd)"/logs,target=/oatomobile/logs \
  #      --user $(id -u):$(id -g) \
  #      dhonerkamp/mycarla:latest ${cmd}
    done
  done

else
  echo "Unknown or missing first argument" && exit 1
fi
