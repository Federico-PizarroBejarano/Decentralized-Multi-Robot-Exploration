import numpy as np
from imageio import imread

def convert_image(filename):
    map = imread(filename, as_gray = True)
    map[map < 128] = 1
    map[map >= 128] = 0

    np.save(filename.split('.')[0], map)

    return map