import os
import random

import numpy as np
import yaml

from decentralized_exploration.core.constants import SEED, LENGTH, WIDTH
from decentralized_exploration.helpers.generate_pixelmap import generate_pixelmap

random.seed(SEED)

NUM_TRAINING_MAPS = 100

MIN_OBSTACLE_DENSITY = .3
MAX_OBSTACLE_DENSITY = .4

OBSTACLE_DENSITIES = [random.uniform(MIN_OBSTACLE_DENSITY, MAX_OBSTACLE_DENSITY) for i in range(NUM_TRAINING_MAPS)]
MAPPATH = os.getcwd().rsplit('helpers')[0] + 'assets/maps/train/'
os.makedirs(MAPPATH, exist_ok=True)

id_to_density = {}
filepaths = []

config = None
config_path = '/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl/assets/config.yaml'
with open(config_path) as stream:
	config = yaml.load(stream, Loader=yaml.SafeLoader)

for id, OBSTACLE_DENSITY in enumerate(OBSTACLE_DENSITIES):
	pixelmap = generate_pixelmap(LENGTH, WIDTH, OBSTACLE_DENSITY, config['color']['free'], config['color']['obstacle'])
	filename = 'map-{:02d}'.format(id)
	filepath = MAPPATH + filename

	np.save(filepath, pixelmap)
	id_to_density[id] = OBSTACLE_DENSITY
	filepaths.append(filepath)

with open(MAPPATH + 'id_to_density.txt', 'w') as f:
	for k, v in id_to_density.items():
		f.write(str(k) + ':' + str(v) + '\n')

with open(MAPPATH + 'filepaths.txt', 'w') as f:
	for filepath in filepaths:
		f.write("%s.npy\n" % filepath)









