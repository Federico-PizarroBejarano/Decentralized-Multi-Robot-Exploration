import numpy as np
import matplotlib.pyplot as plt

from ..helpers.plotting import plot_map
from .test_field_of_view import test_field_of_view
from .test_hex_grid import test_grid_creation
from .test_convert_image_to_pixelmap import test_convert_image_to_pixelmap

if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    # Testing field_of_view
    robot_pos = (30, 77)
    unknown_I = test_field_of_view(I, robot_pos)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_map(unknown_I, plot=ax)

    plt.show()

    # Testing hex_grid
    start = 88
    end = 25

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    grid = test_grid_creation(I, ax1)

    plt.show()

    # Testing convert_image_to_pixelmap
    file_path = './decentralized_exploration/maps/map_1_small.png'
    test_convert_image_to_pixelmap(file_path, I)
