import numpy as np
import matplotlib.pyplot as plt

from ..helpers.generate_pixelmap import generate_pixelmap

LENGTH = 100
WIDTH = 100
OBSTACLE_DENSITY = 0.3

def test_generate_pixelmap():
    pixelmap = generate_pixelmap(LENGTH, WIDTH, OBSTACLE_DENSITY)
    assert(LENGTH, WIDTH) == pixelmap.shape

    area = LENGTH * WIDTH
    num_obstacles = np.count_nonzero(pixelmap)
    print(num_obstacles)

    actual_obstacle_density = num_obstacles / float(area)
    err = 1 / float(area)

    print('Desired obstacle density: {}'.format(OBSTACLE_DENSITY))
    print('Actual obstacle density: {}'.format(actual_obstacle_density))
    assert(abs(actual_obstacle_density - OBSTACLE_DENSITY) <= err)

    plt.imshow(-pixelmap, cmap='gray')
    plt.show()


def main():
    test_generate_pixelmap()


if __name__ == '__main__':
    main()
