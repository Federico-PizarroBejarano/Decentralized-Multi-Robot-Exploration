import numpy as np
import matplotlib.pyplot as plt

from ..helpers.plotting import plot_grid
from ..helpers.grid import convert_pixelmap_to_grid
from ..helpers.field_of_view import field_of_view

from ..helpers.RobotState import RobotState
from ..helpers.decision_making import calculate_dist, calculate_coord, calculate_utility


def test_grid_creation(I, ax1):
    grid = convert_pixelmap_to_grid(I)
    return grid

def test_field_of_view(I, unknown_I, robot_pos):
    occupied_points, free_points = field_of_view(I, robot_pos)

    occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
    free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

    unknown_I[occ_rows, occ_cols] = 1
    unknown_I[free_rows, free_cols] = 0

    return unknown_I

def main():
    I = np.load('./decentralized_exploration/maps/test_1.npy')
    unknown_I = -np.ones(I.shape)

    robot_pos = (1, 3)
    unknown_I = test_field_of_view(I, unknown_I, robot_pos)
    robot_pos = (4, 4)
    unknown_I = test_field_of_view(I, unknown_I, robot_pos)
    robot_pos = (5, 6)
    unknown_I = test_field_of_view(I, unknown_I, robot_pos)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    grid = test_grid_creation(unknown_I, ax1)
    robot_states = []#[RobotState((5, 6))]
    # calculate_dist(grid.all_cells[(0, 0)], grid, 1, 2)
    calculate_utility(grid.all_cells[(0, 0)], grid, robot_states, 1, 2)
    plot_grid(grid, plot=ax1, robot_states={}, mode='utility')

    plt.show()


if __name__ == '__main__':
    main()
