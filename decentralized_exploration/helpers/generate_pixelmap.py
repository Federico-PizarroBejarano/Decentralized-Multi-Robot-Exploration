import numpy as np

from decentralized_exploration.core.constants import SEED

np.random.seed(SEED)

def generate_pixelmap(length, width, obstacle_density, unoccupied_val, occupied_val):
    pixelmap = np.random.rand(length, width)

    threshold = np.percentile(pixelmap, obstacle_density * 100)

    pixelmap[pixelmap > threshold] = unoccupied_val # unoccupied
    pixelmap[pixelmap.nonzero()] = occupied_val # occupied
    
    return pixelmap
