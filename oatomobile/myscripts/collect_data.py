import argparse
from pathlib import Path
import os
import datetime

from carla import WeatherParameters
from oatomobile.myscripts.MapillaryDataset import MapillaryDataset

from absl import logging
logging.set_verbosity(logging.DEBUG)
os.environ["WANDB_API_KEY"] = "61679940976449c2d6a6c842ead1dc8bd975da80"
import wandb


class TownsConfig:
    """https://carla.readthedocs.io/en/latest/core_map/"""
    occupancy = dict()
    occupancy['empty'] = {'num_vehicles': 0,
                          'num_pedestrians': 0}
    occupancy['busyV0'] = {'num_vehicles': 100,
                            'num_pedestrians': 100}
    # towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']
    towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    weather = {
        "ClearNoon": WeatherParameters.ClearNoon,
        "ClearSunset": WeatherParameters.ClearSunset,
        "CloudyNoon": WeatherParameters.CloudyNoon,
        "CloudySunset": WeatherParameters.CloudySunset,
        "Default": WeatherParameters.Default,
        "HardRainNoon": WeatherParameters.HardRainNoon,
        "HardRainSunset": WeatherParameters.HardRainSunset,
        "MidRainSunset": WeatherParameters.MidRainSunset,
        "MidRainyNoon": WeatherParameters.MidRainyNoon,
        "SoftRainNoon": WeatherParameters.SoftRainNoon,
        "SoftRainSunset": WeatherParameters.SoftRainSunset,
        "WetCloudyNoon": WeatherParameters.WetCloudyNoon,
        "WetCloudySunset": WeatherParameters.WetCloudySunset,
        "WetNoon": WeatherParameters.WetNoon,
        "WetSunset": WeatherParameters.WetSunset,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', type=str, default='Town01', choices=TownsConfig.towns)
    parser.add_argument('--occ', type=str, default='busyV0', choices=TownsConfig.occupancy.keys())
    parser.add_argument('--weather', type=str, default='ClearNoon', choices=TownsConfig.weather.keys())
    parser.add_argument('-n', '--nepisodes', type=int, default=1)
    parser.add_argument('--name', type=str, default="", help="Name of the run. If None set to now(). Can be used to reprocess existing folder in combination with -n 0")
    parser.add_argument('--num_steps', type=int, default=1000, help="Steps per episode")
    parser.add_argument('--logdir', type=str, default='/home/honerkam/repos/oatomobile/logs/test')
    parser.add_argument('-c', '--combine_towns', action='store_true')
    parser.add_argument('--wandb_log_n', type=int, default=0, help="Whether to visualise the first n images in wandb")
    parser.add_argument('--process_immediately', action='store_true', help="Directly process after collectin")
    args = parser.parse_args()
    return args


def main(town: str, weather: WeatherParameters, nepisodes, occupancy: str, num_steps: int, logdir: str, run_name: str, do_process: bool = False,
         wandb_log_n: int = 0):
    if not run_name:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(logdir) / '_'.join([town, occupancy, weather]) / run_name
    print("\n######################################################################################################")
    print("# Collecting data. Root path: {}, collecting {} epsiodes, {} steps".format(root, nepisodes, num_steps))
    print("######################################################################################################\n")

    dataset_dir = root / 'raw'
    processed_dir = root / 'processed'
    # e = Episode(output_dir, '4d07f65c76114dafb3dacd055e121269')

    occ = TownsConfig.occupancy[occupancy]
    weath = TownsConfig.weather[weather]

    sensors = (
                  "acceleration",
                  "velocity",
                  # "lidar",
                  "is_at_traffic_light",
                  "traffic_light_state",
                  "actors_tracker",
                  # added by me
                  "front_camera_rgb",
                  # "front_camera_depth",
                  "semantic_lidar"
              )

    for e in range(nepisodes):
        MapillaryDataset.collect(town=town,
                                 output_dir=str(dataset_dir),
                                 num_vehicles=occ['num_vehicles'],
                                 num_pedestrians=occ['num_pedestrians'],
                                 sensors=sensors,
                                 num_steps=num_steps,
                                 render=False,
                                 create_vid=False,
                                 weather=weath)

    if do_process:
        MapillaryDataset.process(dataset_dir=str(dataset_dir),
                                 output_dir=str(processed_dir),
                                 wandb_log_n=wandb_log_n)

    # if do_plot:
    #     f = '081d5439a2a2429a884513473dfbc35b/0fe89c2fd5a04957b289e7e10b66b06e.npz'
    #     CARLADataset.plot_datum(fname='/home/honerkam/repos/oatomobile/logs/autopilot/test_dataset/{}'.format(f),
    #                             output_dir=dataset_dir / 'plots')


def combine_towns(logdir: str, wandb_log_n: int = 0):
    print("Combining all runs found in folder {}".format(logdir))
    MapillaryDataset.show_lengths(logdir)

    MapillaryDataset.process(dataset_dir=logdir,
                             output_dir=os.path.join(logdir, "combined", "processed"),
                             wandb_log_n=wandb_log_n)


if __name__ == '__main__':
    args = get_args()

    if args.wandb_log_n:
        wandb.init(project="carla_efficientps_data")

    if args.combine_towns:
        combine_towns(args.logdir, wandb_log_n=args.wandb_log_n)
    else:
        main(town=args.town,
             nepisodes=args.nepisodes,
             occupancy=args.occ,
             num_steps=args.num_steps,
             logdir=args.logdir,
             run_name=args.name,
             weather=args.weather,
             wandb_log_n=args.wandb_log_n,
             do_process=args.process_immediately)


