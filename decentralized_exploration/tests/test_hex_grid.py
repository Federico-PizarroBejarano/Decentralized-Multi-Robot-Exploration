import numpy as np
import matplotlib.pyplot as plt

from ..helpers.plotting import plot_grid
from ..helpers.hex_grid import convert_pixelmap_to_grid


def test_grid_creation(I, ax1):
    grid = convert_pixelmap_to_grid(I, 7)
    plot_grid(grid, plot=ax1)
    return grid


if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    grid = test_grid_creation(I, ax1)

    plt.show()
