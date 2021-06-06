import numpy as np
import matplotlib.pyplot as plt

import decentralized_exploration.tests.test_field_of_view as test_field_of_view
import decentralized_exploration.tests.test_hex_grid as test_hex_grid
import decentralized_exploration.tests.test_convert_image_to_pixelmap as test_convert_image_to_pixelmap
import decentralized_exploration.tests.test_decision_making as test_decision_making

if __name__ == "__main__":
    test_convert_image_to_pixelmap.main()
    test_field_of_view.main()
    test_hex_grid.main()
    test_decision_making.main()
