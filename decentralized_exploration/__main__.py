import numpy as np

from decentralized_exploration.core.World import World
from decentralized_exploration.core.RangeFinder import RangeFinder
from decentralized_exploration.core.Robot import Robot
from decentralized_exploration.core.RobotTeam import RobotTeam

from decentralized_exploration.helpers.plotting import plot_grid
from decentralized_exploration.helpers.RobotState import RobotState


if __name__ == "__main__":
    world_map = np.load('./decentralized_exploration/maps/map_1_small.npy')

    starting_pos = [30, 77]
    range_finder = RangeFinder(full_range=10, frequency=0.7)

    robot_1 = Robot(robot_id="robot_1", range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
    robot_1_state = RobotState([30, 77], 6)

    robot_2 = Robot(robot_id="robot_2", range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
    robot_2_state = RobotState([77, 30], 3)

    robot_team = RobotTeam()
    robot_team.add_robot(robot_1)
    robot_team.add_robot(robot_2)

    robot_states = {robot_1.robot_id: robot_1_state, robot_2.robot_id: robot_2_state}

    world = World(world_map=world_map, pixel_size=0.02, robot_states=robot_states)

    robot_team.explore(world=world)
