import numpy as np

from ..helpers.plotting import plot_grid, plot_path
from ..helpers.hex_grid import convert_pixelmap_to_grid

def test_grid_creation(I):
    grid = convert_pixelmap_to_grid(I, 7)
    plot_grid(grid)
    return grid

def test_path(grid, start, end):
    plot_path(grid, start, end)

if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')
    
    grid = test_grid_creation(I)
    test_path(grid, 88, 25)