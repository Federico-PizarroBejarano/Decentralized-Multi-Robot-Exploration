import numpy as np

from ..helpers.plotting import plot_map
from ..helpers.field_of_view import field_of_view


def test_field_of_view(I, robot_pos):
    occupied_points, free_points = field_of_view(I, robot_pos)
    unknown_I = -np.ones(I.shape)

    occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
    free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

    unknown_I[occ_rows, occ_cols] = 1
    unknown_I[free_rows, free_cols] = 0

    plot_map(unknown_I)


if __name__ == "__main__":
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')
    robot_pos = (30, 77)
    test_field_of_view(I, robot_pos)
