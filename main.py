import numpy as np
import matplotlib.pyplot as plt
from core import world, range_finder, robot

map = np.load('./maps/map_1.npy')

range_finder = range_finder.RangeFinder(10, 0.7)
small_robot = robot.Robot(range_finder, 20, 20, map.shape)
world = world.World(map, small_robot, [100, 100], 1)

final_map = small_robot.explore(world)

plt.imshow(final_map, cmap = 'gray')
plt.show()