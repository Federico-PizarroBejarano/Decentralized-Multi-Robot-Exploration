import numpy as np

from decentralized_exploration.core.constants import OCCUPIED, UNOCCUPIED, SEED

np.random.seed(SEED)

def generate_pixelmap(length, width, obstacle_density):
    pixelmap = np.random.rand(length, width)

    threshold = np.percentile(pixelmap, obstacle_density * 100)

    pixelmap[pixelmap > threshold] = UNOCCUPIED # unoccupied
    pixelmap[pixelmap.nonzero()] = OCCUPIED # occupied
    
    return pixelmap
