import numpy as np

from .test_field_of_view import test_field_of_view
from .test_hex_grid import test_grid_creation, test_path
from .test_convert_image_to_pixelmap import test_convert_image_to_pixelmap

if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    # Testing field_of_view
    robot_pos = (30, 77)
    test_field_of_view(I, robot_pos)

    # Testing hex_grid
    start = 88
    end = 25

    grid = test_grid_creation(I)
    test_path(grid, start, end)

    # Testing convert_image_to_pixelmap
    file_path = './decentralized_exploration/maps/map_1_small.png'
    test_convert_image_to_pixelmap(file_path, I)
