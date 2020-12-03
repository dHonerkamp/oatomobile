import os
import subprocess
import random
from absl import logging
logging.set_verbosity(logging.DEBUG)

# os.environ["CARLA_ROOT"] = "/home/honerkam/repos/carla"
# os.environ["PYTHONPATH"] = "/home/honerkam/repos/oatomobile:$PYTHONPATH"

"""
export CARLA_ROOT=/home/honerkam/repos/carla
export PYTHONPATH=/home/honerkam/repos/oatomobile:$PYTHONPATH
"""

from oatomobile.myscripts.dataCollection.MapillaryDataset import MapillaryDataset


class NoCrashTrainingConfig:
    """
    https://arxiv.org/pdf/1904.08980.pdf

    "Each episode last from 1 to 5 minutes partitioned in simulation steps of 100 ms." (i.e. 10 Hz)
    -> between 1*60*10=600 and 5*60*10=3000 steps
    -> 10hrs of training data: 10hrs * 3600s * 10hz = 360 000 frames

    Adding noise to 20% of training trajectories with triangular shape
    -> we chose random actions with 10% probability

    We collect data at 20Hz / each 0.05 seconds. Then use frame_skip=5 during processing
    -> 1200 to 6000 steps gives the same length of 1 to 5 minutes
    -> 720 000 frames for same amount of training data -> less is probably fine
    Not implemented:

    """
    towns = ['Town01']
    occupancy = ['noCrash']
    weather = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']


def main():
    # towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
    # towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    # weathers = ['ClearNoon', 'MidRainyNoon', 'CloudySunset']
    # occs = ['busyV0']

    towns = NoCrashTrainingConfig.towns
    weathers = NoCrashTrainingConfig.weather
    occs = NoCrashTrainingConfig.occupancy
    n_settings = len(towns) * len(weathers) * len(occs)

    total_steps = 200000  # overall, not per config combination
    num_steps_per_setting = total_steps // n_settings

    min_steps_per_episode = 5999
    max_steps_per_episode = 6000
    delete_below_length = 1000
    logdir = "/home/honerkam/repos/oatomobile/logs/data_noCrashTrain_v3"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    cmd = """python /home/honerkam/repos/oatomobile/oatomobile/myscripts/dataCollection/collect_data.py 
             --action collect 
             -n 1 
             --num_steps {num_steps} 
             --logdir {logdir} 
             --town {town} 
             --occ {occ} 
             --weather {weather}"""

    cfg = {'towns': towns, 'weathers': weathers, 'occs': occs, 'logdir': logdir, 'min_steps_per_episode': min_steps_per_episode, 'max_steps_per_episode': max_steps_per_episode}
    wandb.init(entity='wazzup', project='carla_data', config=cfg)


    for town in towns:
        for weather in weathers:
            for occ in occs:
                print("Next: {}, {}".format(town, weather, occ))

                while True:
                    stats = MapillaryDataset.show_lengths(logdir, verbose=True, delete_shorter_than=delete_below_length)
                    relevant_folders, collected_steps, total_collected = [], 0, 0
                    for k, v in stats.items():
                        total_collected += v
                        if (town in k) and (weather in k):
                            relevant_folders.append(k)
                            collected_steps += v
                    wandb.log({'total_collected': total_collected})

                    if collected_steps >= num_steps_per_setting:
                        break
                    else:
                        print("\nCollected only {} / {} steps. Trying again.".format(collected_steps, num_steps_per_setting))
                        remaining_steps = num_steps_per_setting - collected_steps
                        # os.removedirs("blub")

                        s = random.randint(min_steps_per_episode, max_steps_per_episode)
                        curr = cmd.format(num_steps=min(max(remaining_steps, delete_below_length), s),
                                          logdir=logdir,
                                          weather=weather,
                                          town=town,
                                          occ=occ)
                        print("\nRunning cmd: {}".format(curr))
                        subprocess.call(curr.split())


if __name__ == '__main__':
    main()
