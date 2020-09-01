from pathlib import Path
from typing import Sequence
import argparse
import os
from pathlib import Path
import numpy as np
import os
import logging
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import umsgpack
import datetime

from oatomobile.core.dataset import Episode, tokens
from oatomobile.datasets.carla import CARLADataset
from oatomobile.utils.carla import LABEL_COLORS


class MapillaryDataset(CARLADataset):
    """https://github.com/mapillary/seamseg/wiki/Seamless-Scene-Segmentation-dataset-format"""

    def __init__(
            self,
            id: str,
    ) -> None:
        super(MapillaryDataset, self).__init__(id)

    @staticmethod
    def process(
            dataset_dir: str,
            output_dir: str,
    ) -> None:

        os.makedirs(output_dir, exist_ok=True)
        d = Path(output_dir)
        for folder in ["img", "depth", "msk", "lst", "coco", "semantic"]:
            os.makedirs(str(d / folder), exist_ok=True)

        # carla definitions
        carla_categories = [ "None", "Buildings", "Fences", "Other", "Pedestrians", "Poles", "RoadLines", "Roads", "Sidewalks", "Vegetation", "Vehicles", "Walls", "TrafficSigns", "sky", "ground", "bridge"]
        carla_palette = LABEL_COLORS.tolist()
        carla_catids = list(range(len(carla_categories)))
        carla_cat2catid = dict(zip(carla_categories, carla_catids))
        carla_cat2color = dict(zip(carla_categories, carla_palette))
        # using the sum of the colors as shortcut for the mapping
        carla_cat2colorsum = dict(zip(carla_categories, [sum(c) for c in carla_palette]))
        # carla_colorsum2cat = dict(zip([sum(c) for c in carla_palette], carla_categories))

        # mapillary definitions
        stuff = ["Buildings", "Fences", "Other", "Poles", "RoadLines", "Roads", "Sidewalks", "Vegetation", "Walls", "TrafficSigns", "sky", "ground", "bridge"]
        things = ["Pedestrians", "Vehicles"]
        # without "None" category
        all_cats = stuff + things
        all_cats_incl_none = ["None"] + all_cats

        mapillary_cat2catid = dict(zip(all_cats_incl_none, range(len(all_cats_incl_none))))
        # carla_catid2mapillary_catid = {carla_cat2catid[c]: mapillary_cat2catid[c] for c in all_cats_incl_none}
        carla_colorsum2mapillary_catid = {carla_cat2colorsum[c]: mapillary_cat2catid[c] for c in all_cats_incl_none}

        meta = {"meta": {
                    "categories": all_cats,
                    "num_stuff": len(stuff),
                    "num_thing": len(things),
                    "palette": [carla_cat2color[c] for c in all_cats],
                    # TODO: is this how I'm supposed to use this?
                    "original_ids": [carla_cat2catid[c] for c in all_cats]
                },
            "images" : []
        }

        # Iterate over all episodes.
        for episode_token in tqdm.tqdm(os.listdir(dataset_dir)):
            logging.debug("Processes {} episode".format(episode_token))
            # Initializes episode handler.
            episode = Episode(parent_dir=dataset_dir, token=episode_token)
            # Fetches all `.npz` files from the raw dataset.
            sequence = episode.fetch()

            for s in sequence:
                observation = episode.read_sample(sample_token=s)
                rgb = observation['front_camera_rgb']
                semantic_lidar = observation['semantic_lidar']
                semantic_camera_obs = semantic_lidar[:, :, :3]
                instance_obs = semantic_lidar[:, :, 3]
                depth = semantic_lidar[:, :, 4:]

                semantic_1d = (255 * semantic_camera_obs.sum(2)).astype(np.int)
                semantic_mapillary_catids = np.vectorize(carla_colorsum2mapillary_catid.get)(semantic_1d)

                # merge with instance mask: replace everything that is car or pedestrian (categories 4, 10) with the id from the instance mask
                segment_mask = semantic_mapillary_catids.copy()
                thing_mask = np.isin(semantic_mapillary_catids, [mapillary_cat2catid["Pedestrians"], mapillary_cat2catid["Vehicles"]])
                segment_mask[thing_mask] = 1000 + instance_obs[thing_mask]
                # correct for those that are originally 0 in instance_obs[thing_mask]
                segment_mask[segment_mask == 1000] = 0

                # map to a [0, n_instances] range
                # sort to make sure the linear id 0 (nothing) will always be mapped to the instance_id 0
                segment_ids = sorted(np.unique(segment_mask))
                # n_instances = len(instance_ids)
                to_linear_segment_id_map = {inst_id: i for i, inst_id in enumerate(segment_ids)}
                segment_mask_linear = np.vectorize(to_linear_segment_id_map.get)(segment_mask)


                # # map to a [0, n_instances] range
                # # sort to make sure the linear id 0 (nothing) will always be mapped to the instance_id 0
                # instance_ids = sorted(np.unique(instance_obs))
                # # n_instances = len(instance_ids)
                # to_linear_instance_id_map = {inst_id: i for i, inst_id in enumerate(instance_ids)}
                # instance_obs_linear = np.vectorize(to_linear_instance_id_map.get)(instance_obs)

                img_meta = MapillaryDataset._save_png(output_dir=d, image_id=s,
                                           rgb=rgb, depth=depth, segment=segment_mask_linear, semantic=semantic_mapillary_catids)
                meta["images"].append(img_meta)

        with open(str(d / "metadata.bin"), "wb") as fid:
            umsgpack.dump(meta, fid, encoding="utf-8")

    @staticmethod
    def _save_png(output_dir, image_id, segment, semantic, depth, rgb):
        fname = "{}.png".format(image_id)

        rgb_rescaled = (255 * rgb).astype(np.uint8)
        rgb_im = Image.fromarray(rgb_rescaled, "RGB")
        rgb_im.save(output_dir / "img" / fname)

        # TODO: scale into an appropriate max_depth range
        depth1d = depth[:, :, 0]  # all channels have the same value
        depth_rescaled = (65536 * depth1d).astype(np.uint16)
        depth_im = Image.fromarray(depth_rescaled.T, "I;16")
        depth_im.save(output_dir / "depth" / fname)

        # multiply by smth like 10000 to actually see something
        msk_im = Image.fromarray(segment.T.astype(np.uint16), "I;16")
        msk_im.save(output_dir / "msk" / fname)

        # comment out, not actually needed
        # semantic_im = Image.fromarray(10000 * semantic.T.astype(np.uint16), "I;16")
        # semantic_im.save(output_dir / "semantic" / fname)

        # for each segment, find what class it is
        vec = np.reshape(np.stack([segment.astype(int), semantic], axis=2), [-1, 2])
        segment_semantic_pairs, counts = np.unique(vec, axis=0, return_counts=True)
        # we might have multiple classes within a single segment mask -> take the one with the highest count
        cats = []
        for s in sorted(set(segment_semantic_pairs[:, 0])):
            m = segment_semantic_pairs[:, 0] == s
            cls = segment_semantic_pairs[m]
            n = counts[m]
            more_freq = int(cls[np.argmax(n)][1])
            cats.append(more_freq)

        # segment_semantic_pairs_sorted = sorted(segment_semantic_pairs, key=lambda x: x[0])
        # cats = [e[1] for e in segment_semantic_pairs_sorted]
        assert cats[0] == 0
        cats[0] = 255


        images_meta = {"id": str(image_id),
                "size": tuple(segment.T.shape),  # (height, width)
                "cat": cats,
                # TODO: is this relevant to know?
                "iscrowd": [1] + (len(cats) - 1) * [0]
        }

        return images_meta


class TownsConfig:
    """https://carla.readthedocs.io/en/latest/core_map/"""
    occupancy = dict()
    occupancy['empty'] = {'num_vehicles': 0,
                          'num_pedestrians': 0}
    occupancy['busy_v0'] = {'num_vehicles': 100,
                            'num_pedestrians': 100}
    towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--town', type=str, default='Town01', choices=TownsConfig.towns)
    parser.add_argument('--occ', type=str, default='busy_v0', choices=TownsConfig.occupancy.keys())
    parser.add_argument('-n', '--nepisodes', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=1000, help="Steps per episode")
    args = parser.parse_args()
    return args


def main(do_collect: bool, do_plot: bool, town: str, nepisodes, occupancy: str, num_steps: int):
    now = datetime.datetime.now()
    root = Path('/home/honerkam/repos/oatomobile/logs/autopilot') / (town + '_' + occupancy) / str(now)

    dataset_dir = root / 'raw'
    processed_dir = root / 'processed'
    # e = Episode(output_dir, '4d07f65c76114dafb3dacd055e121269')

    occ = TownsConfig.occupancy[occupancy]

    if do_collect:
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
                                 create_vid=False)

        MapillaryDataset.process(dataset_dir=str(dataset_dir),
                                 output_dir=str(processed_dir))

    if do_plot:
        f = '081d5439a2a2429a884513473dfbc35b/0fe89c2fd5a04957b289e7e10b66b06e.npz'
        CARLADataset.plot_datum(fname='/home/honerkam/repos/oatomobile/logs/autopilot/test_dataset/{}'.format(f),
                                output_dir=dataset_dir / 'plots')


if __name__ == '__main__':
    args = get_args()
    main(do_collect=True,
         do_plot=False,
         town=args.town,
         nepisodes=args.nepisodes,
         occupancy=args.occ,
         num_steps=args.num_steps)
