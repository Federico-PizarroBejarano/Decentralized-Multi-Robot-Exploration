import numpy as np
import matplotlib.pyplot as plt

from ..helpers.plotting import plot_grid, plot_path
from ..helpers.hex_grid import convert_pixelmap_to_grid


def test_grid_creation(I, ax1):
    grid = convert_pixelmap_to_grid(I, 7)
    plot_grid(grid, plot=ax1)
    return grid


def test_path(grid, start, end, ax2):
    plot_path(grid, start, end, plot=ax2)


if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    grid = test_grid_creation(I, ax1)
    test_path(grid, 88, 25, ax2)

    plt.show()
