import numpy as np

from ..helpers.convert_image_to_pixelmap import convert_image_to_pixelmap

def test_convert_image_to_pixelmap(file_path, pixel_map):
    calculated_map = convert_image_to_pixelmap(file_path)

    if np.all(calculated_map == pixel_map):
        print(True)
    else:
        raise ValueError()

if __name__ == "__main__":
    file_path = './decentralized_exploration/maps/map_1_small.png'
    pixel_map = np.load('./decentralized_exploration/maps/map_1_small.npy')
    test_convert_image_to_pixelmap(file_path, pixel_map)