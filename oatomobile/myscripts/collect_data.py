from typing import Sequence

from oatomobile.core.dataset import Episode
from oatomobile.datasets.carla import CARLADataset

def main(do_collect: bool, do_plot: bool):
    output_dir = '/home/honerkam/repos/oatomobile/logs/autopilot/test_dataset'
    # e = Episode(output_dir, '4d07f65c76114dafb3dacd055e121269')

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
        CARLADataset.collect(town="Town01",
                             output_dir=output_dir,
                             num_vehicles=75,
                             num_pedestrians=25,
                             sensors=sensors,
                             render=False,
                             create_vid=True)

    if do_plot:
        f = '081d5439a2a2429a884513473dfbc35b/0fe89c2fd5a04957b289e7e10b66b06e.npz'
        CARLADataset.plot_datum(fname='/home/honerkam/repos/oatomobile/logs/autopilot/test_dataset/{}'.format(f),
                                output_dir=output_dir + '/plots')


if __name__ == '__main__':
    main(do_collect=True,
         do_plot=False)