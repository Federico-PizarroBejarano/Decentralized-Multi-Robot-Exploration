from imageio import imread


def convert_image_to_pixelmap(file_path):
    """
    Converts an image file into a numpy array of pixel intensities representing free 
    and occupied space returns it

    Parameters
    ----------
    file_path (str): the full path to the desired file, relative to directory where this is run

    Returns
    -------
    pixel_map (numpy.ndarray): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    """

    pixel_map = imread(file_path, as_gray=True)
    pixel_map[pixel_map < 128] = 1
    pixel_map[pixel_map >= 128] = 0

    return pixel_map
