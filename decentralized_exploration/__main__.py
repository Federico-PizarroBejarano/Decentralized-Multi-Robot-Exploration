import numpy as np
import random
import cPickle as pickle

from decentralized_exploration.core.World import World
from decentralized_exploration.core.RangeFinder import RangeFinder
from decentralized_exploration.core.robots.RobotGreedy import RobotGreedy
from decentralized_exploration.core.RobotTeam import RobotTeam
from decentralized_exploration.helpers.RobotState import RobotState
from decentralized_exploration.helpers.grid import convert_pixelmap_to_grid


if __name__ == "__main__":
    world_map = np.load('./decentralized_exploration/maps/new_maps/test_1.npy')
    completed_grid = convert_pixelmap_to_grid(pixel_map=world_map)

    num_of_robots = 1
    robot_team = RobotTeam(world_size=world_map.shape, blocked_by_obstacles=False)

    print('Starting poses: ', (0, 0))

    robot_states = {}

    for r in range(num_of_robots):
        starting_pos = (0, 0)
        range_finder = RangeFinder(full_range=10, frequency=0.7)

        robot = RobotGreedy(robot_id="robot_" + str(r+1), range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
        robot_state = RobotState(pixel_position=starting_pos)

        robot_team.add_robot(robot)
        robot_states[robot.robot_id] = robot_state

    world = World(world_map=world_map, pixel_size=0.02, robot_states=robot_states)

    robot_team.explore(world=world)
