import numpy as np
from imageio import imread


def convert_image_to_pixelmap(file_path):
    """
    Converts an image file into a numpy array of pixel intensities representing free 
    and occupied space returns it

    Parameters
    ----------
    file_path (str): the path to the desired file

    Returns
    ----------
    numpy.ndarray: numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    """

    map = imread(file_path, as_gray=True)
    map[map < 128] = 1
    map[map >= 128] = 0

    return map
