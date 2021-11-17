import numpy as np
import pickle
import os

from decentralized_exploration.core.World import World
from decentralized_exploration.core.RangeFinder import RangeFinder
from decentralized_exploration.core.robots.RobotGreedy import RobotGreedy
from decentralized_exploration.core.RobotTeam import RobotTeam
from decentralized_exploration.helpers.RobotState import RobotState
from decentralized_exploration.helpers.generate_pixelmap import generate_pixelmap
from decentralized_exploration.helpers.grid import convert_pixelmap_to_grid


if __name__ == '__main__':
    all_starting_poses = {  
                            'top_left':[(0, 0), (1, 0), (0, 1)], 
                            'top_right':[(0, 19), (1, 19), (0, 18)], 
                            'bottom_left':[(19, 0), (18, 0), (19, 1)], 
                            'bottom_right':[(19, 19), (18, 19), (19, 18)] 
                        }

    algorithms = [
                    'greedy',
                ]

    all_files = os.listdir('./decentralized_exploration/results')

    length, width, object_density = 20, 20, 0.3

    for algorithm in algorithms:
        for test in range(1):
            for starting_poses_key in all_starting_poses.keys():
                for pfc in [0, 100]:
                    for fci in [7]:
                        filename = '{}_{}_{}_{}fc_{}iters_rerun.pkl'.format(algorithm, test, starting_poses_key, pfc, fci)

                        if filename in all_files:
                            print('{}_{}_{}_{}fc_{}iters_rerun.pkl'.format(algorithm, test, starting_poses_key, pfc, fci) + ' SKIPPED!!')
                            continue

                        print(algorithm, test, starting_poses_key, '{}% fail'.format(pfc), fci)
                        
                        if os.path.isfile('./decentralized_exploration/maps/test_{}.npy'.format(test)): 
                            world_map = np.load('./decentralized_exploration/maps/test_{}.npy'.format(test))

                        else:
                            world_map = generate_pixelmap(length, width, object_density)
                            # ensure the robots starting positions are empty
                            for start_positions in all_starting_poses.values():
                                for start_position in start_positions:
                                    world_map[start_position[0], start_position[1]] = 0
                            np.save('./decentralized_exploration/maps/test_0', world_map)


                        completed_grid = convert_pixelmap_to_grid(pixel_map=world_map)

                        num_of_robots = 3
                        robot_team = RobotTeam(world_size=world_map.shape, communication_range=4, blocked_by_obstacles=True, failed_communication_interval=fci, probability_of_failed_communication=pfc)

                        starting_poses = all_starting_poses[starting_poses_key]
                        print('Starting poses: ', starting_poses)

                        robot_states = {}

                        for r in range(num_of_robots):
                            starting_pos = starting_poses[r]
                            range_finder = RangeFinder(full_range=10, frequency=0.7)

                            if algorithm == 'greedy':
                                robot = RobotGreedy(robot_id='robot_' + str(r+1), range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
                            
                            robot_state = RobotState(pixel_position=starting_pos)

                            robot_team.add_robot(robot)
                            robot_states[robot.robot_id] = robot_state

                        world = World(world_map=world_map, pixel_size=1, robot_states=robot_states)

                        data = robot_team.explore(world=world)

                        with open('./decentralized_exploration/results/'+filename, 'wb') as outfile:
                            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
