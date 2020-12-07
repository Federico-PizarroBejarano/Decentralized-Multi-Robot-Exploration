import numpy as np

from core import world, range_finder, robot
from helpers.plotting import plot_grid

world_map = np.load('./maps/map_1_small.npy')

starting_pos = [30, 77]
range_finder = range_finder.RangeFinder(10, 0.7)
small_robot = robot.Robot(range_finder, 20, 20, world_map.shape)
world = world.World(world_map, 0.02, small_robot, starting_pos, 1)

final_map = small_robot.explore(world)