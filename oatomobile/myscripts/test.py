from oatomobile.core.dataset import Episode
from oatomobile.core.dataset import tokens


if __name__ == '__main__':
    output_dir = '/home/honerkam/repos/oatomobile/logs/autopilot/test'
    e = Episode(output_dir, '4d07f65c76114dafb3dacd055e121269')

    sequence = e.fetch()
    obs = []
    for s in sequence:
        o = e.read_sample(s)
        obs.append(o)

    for k, v in o.items():
        print(k, v.shape)

    print('blub')