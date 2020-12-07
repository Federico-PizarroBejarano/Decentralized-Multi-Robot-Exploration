import numpy as np

from .test_field_of_view import test_field_of_view
from .test_hex_grid import test_grid_creation, test_path

if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    robot_pos = (30, 77)
    test_field_of_view(I, robot_pos)

    start = 88
    end = 25

    grid = test_grid_creation(I)
    test_path(grid, start, end)
