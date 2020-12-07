import numpy as np

from ..helpers.plotting import plot_grid, plot_path
from ..helpers.hex_grid import convert_image_to_grid

if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')
    grid = convert_image_to_grid(I, 7)
    plot_grid(grid)
    plot_path(grid, 88, 25)