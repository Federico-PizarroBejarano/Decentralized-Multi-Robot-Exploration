import numpy as np

from decentralized_exploration.core import world, range_finder, robot
from decentralized_exploration.helpers.plotting import plot_grid

if __name__ == "__main__":
    world_map = np.load('./decentralized_exploration/maps/map_1_small.npy')

    starting_pos = [30, 77]
    range_finder = range_finder.RangeFinder(10, 0.7)
    small_robot = robot.Robot(range_finder, 20, 20, world_map.shape)
    world = world.World(world_map, 0.02, small_robot, starting_pos, 1)

    final_map = small_robot.explore(world)
