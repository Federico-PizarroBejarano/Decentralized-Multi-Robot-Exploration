import numpy as np
import matplotlib.pyplot as plt

from ..helpers.convert_image_to_pixelmap import convert_image_to_pixelmap


def test_convert_image_to_pixelmap(file_path, pixel_map):
    calculated_map = convert_image_to_pixelmap(file_path)

    if np.all(calculated_map == pixel_map):
        plt.imshow(-calculated_map, cmap='gray')
        plt.show()
    else:
        raise ValueError()


def main():
    file_path = './decentralized_exploration/maps/map_1_small.png'
    pixel_map = np.load('./decentralized_exploration/maps/map_1_small.npy')
    test_convert_image_to_pixelmap(file_path, pixel_map)


if __name__ == '__main__':
    main()
