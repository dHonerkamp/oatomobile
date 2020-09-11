import logging
import os
from pathlib import Path

import numpy as np
import tqdm
import umsgpack
from PIL import Image
import wandb

from oatomobile import Episode
from oatomobile.datasets import CARLADataset
from oatomobile.utils.carla import LABEL_COLORS


class MapillaryDataset(CARLADataset):
    """https://github.com/mapillary/seamseg/wiki/Seamless-Scene-Segmentation-dataset-format"""

    def __init__(
            self,
            id: str,
    ) -> None:
        super(MapillaryDataset, self).__init__(id)

    @staticmethod
    def get_episode_dirs(dataset_dir: str):
        # output dir is either directly a dataset folder or the overall log folder with many towns
        dirs = []
        for d in os.listdir(dataset_dir):
            if "Town" in d:
                runs = os.listdir(os.path.join(str(dataset_dir), d))
                for r in runs:
                    episode_folder = os.path.join(dataset_dir, d, r, "raw")
                    if not os.path.exists(episode_folder):
                        print("Skipping non-existing folder: {}".format(episode_folder))
                        continue
                    else:
                        dirs += [os.path.join(episode_folder, e) for e in os.listdir(episode_folder)]

            else:
                if dataset_dir.split('/')[-1] == "raw":
                    dirs.append(d)
                else:
                    print("Skipping folder {}".format(d))
                    continue
        return sorted(dirs)

    @staticmethod
    def show_lengths(dataset_dir, verbose=True):
        stats = {}

        dirs = MapillaryDataset.get_episode_dirs(dataset_dir)
        for episode_token in dirs:
            logging.debug("Processes {} episode".format(episode_token))
            # Initializes episode handler.
            episode = Episode(parent_dir=dataset_dir, token=episode_token)
            # Fetches all `.npz` files from the raw dataset.
            try:
                sequence = episode.fetch()
                if verbose:
                    print("Folder: {}".format(episode_token))
                    print("Sequence tokens: {}".format(len(sequence)))
                stats[episode_token] = len(sequence)
            except Exception as e:
                if verbose:
                    print("Skipping folder {}: {}".format(episode_token, e))
                stats[episode_token] = 0
                continue
        return stats

    @staticmethod
    def process(
            dataset_dir: str,
            output_dir: str,
            split: list = [0.6, 0.2, 0.2],
            wandb_log_n: int = 0,
    ) -> None:

        assert sum(split) == 1, "Split doesn't sum to 1"

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        for folder in ["img", "depth", "msk", "lst", "coco", "semantic"]:
            os.makedirs(str(output_dir / folder), exist_ok=True)

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


        # output dir is either directly a dataset folder or the overall log folder with many towns
        dirs = MapillaryDataset.get_episode_dirs(dataset_dir)
        wandb_imgs = []

        # Iterate over all episodes.
        for episode_token in dirs:
            logging.debug("Processes {} episode".format(episode_token))
            # Initializes episode handler.
            episode = Episode(parent_dir=dataset_dir, token=episode_token)
            # Fetches all `.npz` files from the raw dataset.
            try:
                sequence = episode.fetch()
            except Exception as e:
                print("Skipping folder {}: {}".format(episode_token, e))
                continue

            for s in tqdm.tqdm(sequence):
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

                img_meta, wandb_image = MapillaryDataset._save_png(output_dir=output_dir, image_id=s,
                                                      rgb=rgb, depth=depth, segment=segment_mask_linear, semantic=semantic_mapillary_catids,
                                                      num_stuff=len(stuff), num_thing=len(things), mapillary_class_lbls=all_cats, log_wandb=len(wandb_imgs) < wandb_log_n)
                meta["images"].append(img_meta)


                if len(wandb_imgs) < wandb_log_n:
                    wandb_imgs.append(wandb_image)

        if wandb_log_n:
            # for i, img in enumerate(wandb_imgs):
            #     wandb.log({'data_coll{}'.format(i): img})
            wandb.log({"data_coll": wandb_imgs})

        with open(str(output_dir / "metadata.bin"), "wb") as fid:
            umsgpack.dump(meta, fid, encoding="utf-8")

        n = len(meta["images"])
        n_train = int(split[0] * n)
        n_val = int(split[1] * n)

        train_ids = [img["id"] for img in meta["images"][:n_train]]
        val_ids = [img["id"] for img in meta["images"][n_train:n_train + n_val]]
        test_ids = [img["id"] for img in meta["images"][n_train + n_val:]]

        for name, s in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
            with open(str(output_dir / "lst" / (name + ".txt")), 'w') as f:
                f.writelines('\n'.join(s))

        print("\n######################################################################################################")
        print("# Processed {} images".format(len(meta["images"])))
        print("######################################################################################################\n")

    @staticmethod
    def _save_png(output_dir, image_id, segment, semantic, depth, rgb, num_stuff, num_thing, mapillary_class_lbls, log_wandb: bool):
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
        # segmentid 0 is always the empty class, mapped to category 255
        cats = []
        for s in sorted(set(segment_semantic_pairs[:, 0])):
            if s == 0:
                cats.append(255)
            else:
                m = segment_semantic_pairs[:, 0] == s
                cls = segment_semantic_pairs[m]
                n = counts[m]
                more_freq = int(cls[np.argmax(n)][1])
                # while segmentsIds have 0 == empty, categoryIds start with 0 == first category -> shift left by one
                cats.append(more_freq - 1)

                if s < 11 and more_freq != s:
                    print(s, more_freq)

        # print(image_id)
        # print(segment_semantic_pairs)
        # print(cats)

        # segment_semantic_pairs_sorted = sorted(segment_semantic_pairs, key=lambda x: x[0])
        # cats = [e[1] for e in segment_semantic_pairs_sorted]
        # assert cats[0] == 0, cats
        # cats[0] = 255
        assert cats[0] == 255, cats
        assert max(cats[1:]) < num_stuff + num_thing  # < not <= due to 0-indexing
        assert len(np.unique(segment)) == len(cats), (np.unique(segment), cats)

        images_meta = {"id": str(image_id),
                       "size": (msk_im.size[1], msk_im.size[0]),  # (height, width)????
                       "cat": cats,
                       # TODO: is this relevant to know?
                       "iscrowd": [1] + (len(cats) - 1) * [0]
        }

        if log_wandb:
            # IMPORTANT: SEEMS IT CAN CAUSE INCORRECT MASK LABELS TO BE DISPLAYED ONLINE IF WE HAVE CHANGING CLASS LABELS LIKE WE DO HERE
            mapillary_clsid2cls = {i: c for i, c in enumerate(mapillary_class_lbls)}
            mapillary_clsid2cls[255] = "None"
            # print({i: mapillary_clsid2cls[c] for i, c in enumerate(cats)})

            wandb_image = wandb.Image(rgb,
                caption="Segment {}".format(image_id),
                masks={
                    "segment": {
                        "mask_data": segment.T.astype(np.uint16),
                        "class_labels": {i: str(i) for i in range(len(cats))}
                    },
                    "semanticMsk": {
                        "mask_data": segment.T.astype(np.uint16),
                        # "class_labels": {i: mapillary_clsid2cls[cats[i]] for i in range(len(cats))}
                        "class_labels": {i: mapillary_clsid2cls[c] for i, c in enumerate(cats)}
                    },
                    "semanticOrig": {
                        "mask_data": semantic.T.astype(np.uint16),
                        "class_labels": {i: c for i, c in enumerate(["None"] + mapillary_class_lbls)}
                    },
                })
        else:
            wandb_image = None
        return images_meta, wandb_image
