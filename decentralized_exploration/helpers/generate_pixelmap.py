import numpy as np
import matplotlib.pyplot as plt


def generate_pixelmap(length, width, obstacle_density):
    pixelmap = np.random.rand(length, width)

    threshold = np.percentile(pixelmap, obstacle_density * 100)

    pixelmap[pixelmap > threshold] = 0 # unoccupied
    pixelmap[pixelmap.nonzero()] = 1 # occupied
    
    return pixelmap


if __name__ == '__main__':
    # p = generate_pixelmap(5,5,0.3)
    # print(p)
    p = np.load('/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/maps/test_1.npy')
    plt.imshow(p, cmap='gray')
    plt.show()
