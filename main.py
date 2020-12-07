import numpy as np
import matplotlib.pyplot as plt

from core import world, range_finder, robot
from helpers import hex_grid

world_map = np.load('./maps/map_1_small.npy')

range_finder = range_finder.RangeFinder(10, 0.7)
small_robot = robot.Robot(range_finder, 20, 20, world_map.shape)
world = world.World(world_map, 0.02, small_robot, [30, 77], 1)

final_map = small_robot.explore(world)

plt.imshow(final_map, cmap = 'gray')
plt.show()

hex_grid.plot_grid(small_robot.get_hex_map())