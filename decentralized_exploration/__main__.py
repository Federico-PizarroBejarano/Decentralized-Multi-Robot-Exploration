import numpy as np
import random
import cPickle as pickle

from decentralized_exploration.core.World import World
from decentralized_exploration.core.RangeFinder import RangeFinder
from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.RobotTeam import RobotTeam
from decentralized_exploration.helpers.RobotState import RobotState
from decentralized_exploration.helpers.hex_grid import convert_pixelmap_to_grid

from decentralized_exploration.helpers.plotting import plot_one_set


if __name__ == "__main__":
    world_map = np.load('./decentralized_exploration/maps/large_map_4.npy')
    completed_grid = convert_pixelmap_to_grid(pixel_map=world_map, size=RobotMDP.hexagon_size)
    hexes_near_entrance = completed_grid.hex_neighbours(completed_grid.find_hex(completed_grid.hex_at(point=[500, 300])), radius=5)

    count = 0

    while count < 4:
        try:
            print('Test #', count)
            num_of_robots = 2
            robot_team = RobotTeam(world_size=world_map.shape)
            starting_hexes = random.sample(population=[h for h in hexes_near_entrance if h.state == 0], k=num_of_robots)
            starting_poses = [completed_grid.hex_center(hexagon=h) for h in starting_hexes]

            print('Starting poses: ', starting_poses)

            robot_states = {}

            for r in range(num_of_robots):
                starting_pos = [int(pos) for pos in starting_poses[r]]
                range_finder = RangeFinder(full_range=10, frequency=0.7)

                robot = RobotMDP(robot_id="robot_" + str(r+1), range_finder=range_finder, width=20, length=20, world_size=world_map.shape)
                robot_state = RobotState(starting_pos, np.random.randint(1, 7))

                robot_team.add_robot(robot)
                robot_states[robot.robot_id] = robot_state

            world = World(world_map=world_map, pixel_size=0.02, robot_states=robot_states)

            robot_team.explore(world=world)
            count += 1
        except:
            print("FAILURE!!!!")

    with open('./decentralized_exploration/results/two_robots_map_4/mdp.pkl', 'rb') as infile:
        all_results = pickle.load(infile)
    
    plot_one_set(all_results)
