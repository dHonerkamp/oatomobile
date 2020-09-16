import os
import subprocess
from absl import logging
logging.set_verbosity(logging.DEBUG)

# os.environ["CARLA_ROOT"] = "/home/honerkam/repos/carla"
# os.environ["PYTHONPATH"] = "/home/honerkam/repos/oatomobile:$PYTHONPATH"

"""
export CARLA_ROOT=/home/honerkam/repos/carla
export PYTHONPATH=/home/honerkam/repos/oatomobile:$PYTHONPATH
"""

from oatomobile.myscripts.MapillaryDataset import MapillaryDataset


def main():
    # towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
    towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    weathers = ['ClearNoon', 'MidRainyNoon', 'CloudySunset']
    num_steps = 5000
    logdir = "/home/honerkam/repos/oatomobile/logs/data_v4"

    cmd = """python /home/honerkam/repos/oatomobile/oatomobile/myscripts/collect_data.py 
             --action collect 
             -n 1 
             --num_steps {num_steps} 
             --logdir {logdir} 
             --town {town} 
             --occ busyV0 
             --weather {weather}"""

    for town in towns:
        for weather in weathers:
            print("Next: {}, {}".format(town, weather))
            remaining_steps = num_steps

            while True:
                stats = MapillaryDataset.show_lengths(logdir, verbose=True)
                relevant_folders, collected_steps = [], 0
                for k, v in stats.items():
                    if (town in k) and (weather in k):
                        relevant_folders.append(k)
                        collected_steps += v

                if collected_steps >= num_steps:
                    break
                else:
                    print("\nCollected only {} / {} steps. Trying again.".format(collected_steps, num_steps))
                    remaining_steps = num_steps - collected_steps
                    # os.removedirs("blub")

                    curr = cmd.format(num_steps=remaining_steps,
                                      logdir=logdir,
                                      weather=weather,
                                      town=town)
                    print("\nRunning cmd: {}".format(curr))
                    subprocess.call(curr.split())



if __name__ == '__main__':
    main()
