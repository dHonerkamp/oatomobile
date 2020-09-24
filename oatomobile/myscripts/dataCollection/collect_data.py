import argparse
from pathlib import Path
import os
import datetime
import random

from carla import WeatherParameters
from oatomobile.myscripts.MapillaryDataset import MapillaryDataset, TownsConfig
from oatomobile.myscripts.myCarlaDataset import MyCarlaDataset
from oatomobile.datasets.carla import CARLADataset

from absl import logging
logging.set_verbosity(logging.DEBUG)
os.environ["WANDB_API_KEY"] = "61679940976449c2d6a6c842ead1dc8bd975da80"
import wandb

"""
Info on mapillary dataformat:
- https://github.com/mapillary/seamseg/wiki/Seamless-Scene-Segmentation-dataset-format
- https://cocodataset.org/#format-data
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', type=str, default='Town01', choices=TownsConfig.towns)
    parser.add_argument('--occ', type=str, default='busyV0', choices=TownsConfig.occupancy.keys())
    parser.add_argument('--weather', type=str, default='ClearNoon', choices=TownsConfig.weather.keys())
    parser.add_argument('-n', '--nepisodes', type=int, default=1)
    parser.add_argument('--name', type=str, default="", help="Name of the run. If None set to now(). Can be used to reprocess existing folder in combination with -n 0")
    parser.add_argument('--num_steps', type=int, default=1000, help="Steps per episode")
    parser.add_argument('--logdir', type=str, default='/home/honerkam/repos/oatomobile/logs/data_test')
    parser.add_argument('--action', type=str, default='collect', choices=['collect', 'combine', 'oxford'])
    parser.add_argument('--wandb_log_n', type=int, default=0, help="Whether to visualise the first n images in wandb")
    parser.add_argument('--process_immediately', action='store_true', help="Directly process after collectin")
    parser.add_argument('--val_share', type=float, default=0.1, help="Share of the training data to use for validation")
    parser.add_argument('--test_towns', nargs='+', default=None, choices=TownsConfig.towns, help="Towns to use for the test set")
    args = parser.parse_args()
    return args


def main(town: str, weather: WeatherParameters, nepisodes, occupancy: str, num_steps: int, logdir: str, run_name: str, do_process: bool = False,
         wandb_log_n: int = 0, val_share=None, test_towns=None):
    if not run_name:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    root = Path(logdir) / TownsConfig.get_task_name(town=town, occupancy=occupancy, weather=weather) / run_name
    print("\n######################################################################################################")
    print("# Collecting data. Root path: {}, collecting {} epsiodes, {} steps".format(root, nepisodes, num_steps))
    print("######################################################################################################\n")

    dataset_dir = root / 'raw'
    processed_dir = root / 'processed'
    carla_dataset_dir = root / 'carla_dataset'
    # e = Episode(output_dir, '4d07f65c76114dafb3dacd055e121269')

    occ = TownsConfig.occupancy[occupancy]
    weath = TownsConfig.weather[weather]

    sensors = (
                  "acceleration",
                  "velocity",
                  "control",  # [throttle, steer, brake]
                  "lidar",
                  "is_at_traffic_light",
                  "traffic_light_state",
                  "actors_tracker",
                  # added by me
                  "front_camera_rgb",
                  # "front_camera_depth", -> get from semantic lidar
                  "semantic_lidar"
              )

    for e in range(nepisodes):
        if isinstance(occ['num_vehicles'], (list, tuple)):
            assert len(occ['num_vehicles']) == 2
            num_vehicles = random.randint(occ['num_vehicles'][0], occ['num_vehicles'][1])
        else:
            num_vehicles = occ['num_vehicles']
        if isinstance(occ['num_pedestrians'], (list, tuple)):
            assert len(occ['num_pedestrians']) == 2
            num_pedestrians = random.randint(occ['num_pedestrians'][0], occ['num_pedestrians'][1])
        else:
            num_pedestrians = occ['num_pedestrians']

        MapillaryDataset.collect(town=town,
                                 output_dir=str(dataset_dir),
                                 num_vehicles=num_vehicles,
                                 num_pedestrians=num_pedestrians,
                                 sensors=sensors,
                                 num_steps=num_steps,
                                 render=False,
                                 create_vid=False,
                                 weather=weath)

    if do_process:
        MapillaryDataset.process(dataset_dir=str(dataset_dir),
                                 output_dir=str(processed_dir),
                                 wandb_log_n=wandb_log_n,
                                 val_share=val_share,
                                 test_towns=test_towns)

        MyCarlaDataset.process(dataset_dir=str(dataset_dir),
                               output_dir=str(carla_dataset_dir),
                               mapillary_lst_dir=str(processed_dir / 'lst'))

    # if do_plot:
    #     f = '081d5439a2a2429a884513473dfbc35b/0fe89c2fd5a04957b289e7e10b66b06e.npz'
    #     CARLADataset.plot_datum(fname='/home/honerkam/repos/oatomobile/logs/autopilot/test_dataset/{}'.format(f),
    #                             output_dir=dataset_dir / 'plots')


def combine_towns(logdir: str, wandb_log_n: int = 0, val_share=None, test_towns=None):
    print("Combining all runs found in folder {}".format(logdir))
    MapillaryDataset.show_lengths(logdir)

    processed_dir = os.path.join(logdir, "combined", "processed")
    carla_dataset_dir = os.path.join(logdir, "combined", "carla_dataset")

    MapillaryDataset.process(dataset_dir=logdir,
                             output_dir=processed_dir,
                             wandb_log_n=wandb_log_n,
                             val_share=val_share,
                             test_towns=test_towns)

    MyCarlaDataset.process(dataset_dir=logdir,
                           output_dir=carla_dataset_dir,
                           mapillary_lst_dir=os.path.join(processed_dir, 'lst'))


def download_oxford_data(output_dir):
    dataset = CARLADataset('processed')
    dataset.download_and_prepare(output_dir)


if __name__ == '__main__':
    args = get_args()

    if args.wandb_log_n:
        wandb.init(project="carla_efficientps_data")

    if args.action == 'combine':
        combine_towns(args.logdir, wandb_log_n=args.wandb_log_n, val_share=args.val_share, test_towns=args.test_towns)
    elif args.action == 'oxford':
        download_oxford_data(args.logdir)
    elif args.action == 'collect':
        main(town=args.town,
             nepisodes=args.nepisodes,
             occupancy=args.occ,
             num_steps=args.num_steps,
             logdir=args.logdir,
             run_name=args.name,
             weather=args.weather,
             wandb_log_n=args.wandb_log_n,
             do_process=args.process_immediately,
             val_share=args.val_share,
             test_towns=args.test_towns)
    else:
        raise NotImplementedError("--action {} not implemented".format(args.action))

