import numpy as np
import matplotlib.pyplot as plt

from decentralized_exploration.helpers.decision_making import voronoi_paths


def test_voronoi_paths(I):
    return voronoi_paths(I)


def main():
    I = np.load('./decentralized_exploration/maps/map_1_small.npy')

    points = test_voronoi_paths(I)

    plt.imshow(I, cmap='gray')
    plt.scatter(points[:,1], points[:, 0])

    plt.show()


if __name__ == "__main__":
    main()