import numpy as np
import matplotlib.pyplot as plt

from ..helpers.plotting import plot_map
from decentralized_exploration.core.robots.utils.field_of_view import field_of_view


def test_field_of_view(I, robot_pos):
    occupied_points, free_points = field_of_view(I, robot_pos)
    unknown_I = -np.ones(I.shape)

    occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
    free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

    unknown_I[occ_rows, occ_cols] = 1
    unknown_I[free_rows, free_cols] = 0

    return unknown_I


def main():
    I = np.load('./decentralized_exploration/maps/test_1.npy')
    robot_pos = (1, 3)
    unknown_I = test_field_of_view(I, robot_pos)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_map(unknown_I, plot=ax)

    plt.show()


if __name__ == '__main__':
    main()
