import numpy as np
import random
import cPickle as pickle

from decentralized_exploration.core.World import World
from decentralized_exploration.core.RangeFinder import RangeFinder
from decentralized_exploration.core.robots.RobotGreedy import RobotGreedy
from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.RobotTeam import RobotTeam
from decentralized_exploration.helpers.RobotState import RobotState
from decentralized_exploration.helpers.grid import convert_pixelmap_to_grid


if __name__ == "__main__":
    all_starting_poses = {  
                            'top_left':[(0, 0), (1, 0), (0, 1)], 
                            'top_right':[(0, 19), (1, 19), (0, 18)],
                            'bottom_left':[(19, 0), (18, 0), (19, 1)], 
                            'bottom_right':[(19, 19), (18, 19), (19, 18)] 
                        }

    for test in range(1, 11):
        for starting_poses_key in all_starting_poses.keys():
            print("test", test, starting_poses_key)
            world_map = np.load('./decentralized_exploration/maps/test_{}.npy'.format(test))
            completed_grid = convert_pixelmap_to_grid(pixel_map=world_map)

            num_of_robots = 3
            robot_team = RobotTeam(world_size=world_map.shape, communication_range=4, blocked_by_obstacles=True)

            starting_poses = all_starting_poses[starting_poses_key]
            print('Starting poses: ', starting_poses)

            robot_states = {}

            for r in range(num_of_robots):
                starting_pos = starting_poses[r]
                range_finder = RangeFinder(full_range=10, frequency=0.7)

                # robot = RobotGreedy(robot_id="robot_" + str(r+1), range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
                robot = RobotMDP(robot_id="robot_" + str(r+1), range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
                robot_state = RobotState(pixel_position=starting_pos)

                robot_team.add_robot(robot)
                robot_states[robot.robot_id] = robot_state

            world = World(world_map=world_map, pixel_size=1, robot_states=robot_states)

            data = robot_team.explore(world=world)

            with open('./decentralized_exploration/results/mdp_{}_{}.pkl'.format(test, starting_poses_key), 'wb') as outfile:
                pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
