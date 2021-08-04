import numpy as np
import matplotlib.pyplot as plt

import decentralized_exploration.tests.test_field_of_view as test_field_of_view
import decentralized_exploration.tests.test_grid as test_grid
import decentralized_exploration.tests.test_convert_image_to_pixelmap as test_convert_image_to_pixelmap

if __name__ == "__main__":
    test_convert_image_to_pixelmap.main()
    test_field_of_view.main()
    test_grid.main()
