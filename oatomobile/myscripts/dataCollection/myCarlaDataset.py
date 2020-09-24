import glob
import os
import sys
import zipfile
from typing import Any
from typing import Callable
from typing import Generator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wget
from absl import logging

import carla

from oatomobile.core.dataset import Dataset
from oatomobile.core.dataset import Episode
from oatomobile.datasets import CARLADataset
from oatomobile.myscripts.MapillaryDataset import MapillaryDataset


class MyCarlaDataset(CARLADataset):
    @staticmethod
    def load_split_lookup(mapillary_lst_dir: str) -> dict:
        sequence_lookup = {}

        for split_name in MapillaryDataset.split_names:
            with open(str(Path(mapillary_lst_dir) / (split_name + ".txt")), 'r') as f:
                sequences = set(f.read().splitlines())
                sequence_lookup.update({s: split_name for s in sequences})

        return sequence_lookup

    @staticmethod
    def process(
            dataset_dir: str,
            output_dir: str,
            mapillary_lst_dir: str,
            future_length: int = 80,
            past_length: int = 20,
            num_frame_skips: int = 5,
    ) -> None:

        from oatomobile.utils import carla as cutil

        # Creates the necessary output directory.
        os.makedirs(output_dir, exist_ok=True)
        for split_name in MapillaryDataset.split_names:
            os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        split_lookup = MyCarlaDataset.load_split_lookup(mapillary_lst_dir)

        dirs = MapillaryDataset.get_episode_dirs(dataset_dir)

        # Iterate over all episodes.
        for episode_token in dirs:
            logging.debug("Processes {} episode".format(episode_token))
            # Initializes episode handler.
            episode = Episode(parent_dir=dataset_dir, token=episode_token)
            # Fetches all `.npz` files from the raw dataset.
            try:
                sequence = episode.fetch()
            except FileNotFoundError:
                "File for episode token {} not found, continuing with next token".format(episode_token)
                continue

            # Always keep `past_length+future_length+1` files open.
            if not len(sequence) >= past_length + future_length + 1:
                print("SEQUENCE {} IS TOO SHORT. IGNORING IT AND MOVING ON.".format(sequence))
                continue

            for i in tqdm.trange(
                    past_length,
                    len(sequence) - future_length,
                    num_frame_skips,
                ):
                # Player context/observation.
                observation = episode.read_sample(sample_token=sequence[i])
                current_location = observation["location"]
                current_rotation = observation["rotation"]

                # Build past trajectory.
                player_past = list()
                for j in range(past_length, 0, -1):
                    past_location = episode.read_sample(
                        sample_token=sequence[i - j],
                        attr="location",
                    )
                    player_past.append(past_location)
                player_past = np.asarray(player_past)
                assert len(player_past.shape) == 2
                player_past = cutil.world2local(
                    current_location=current_location,
                    current_rotation=current_rotation,
                    world_locations=player_past,
                )

                # Build future trajectory.
                player_future = list()
                for j in range(1, future_length + 1):
                    future_location = episode.read_sample(
                        sample_token=sequence[i + j],
                        attr="location",
                    )
                    player_future.append(future_location)
                player_future = np.asarray(player_future)
                assert len(player_future.shape) == 2
                player_future = cutil.world2local(
                    current_location=current_location,
                    current_rotation=current_rotation,
                    world_locations=player_future,
                )

                # Store to ouput directory.
                split_name = split_lookup[sequence[i]]

                np.savez_compressed(
                    os.path.join(output_dir, split_name, "{}.npz".format(sequence[i])),
                    **observation,
                    player_future=player_future,
                    player_past=player_past,
                )
