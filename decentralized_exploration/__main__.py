import numpy as np

from decentralized_exploration.core import world, range_finder, robot
from decentralized_exploration.helpers.plotting import plot_grid

if __name__ == "__main__":
    world_map = np.load('./decentralized_exploration/maps/map_1_small.npy')

    starting_pos = [30, 77]
    range_finder = range_finder.RangeFinder(full_range=10, frequency=0.7)
    small_robot = robot.Robot(range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
    world = world.World(world_map=world_map, pixel_size=0.02, robot=small_robot, robot_position=starting_pos, robot_orientation=1)

    final_map = small_robot.explore(world=world)
