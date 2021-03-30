import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from time import time

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.helpers.hex_grid import convert_pixelmap_to_grid, merge_map
from decentralized_exploration.helpers.decision_making import check_distance_to_other_robot
from decentralized_exploration.helpers.plotting import plot_grid

class RobotTeam:
    """
    A class used to represent a team of robots
    
    Class Attributes
    ----------------
    local_interaction_dist (float): the maximum pixel distance between robots to be considered
        a local interaction
    local_interaction_path_length (int): the maximum path length between robots to be considered
        a local interaction

    Instance Attributes
    -------------------
    robots (dict): a dictionary storing the robots using their robot_ids as keys
    communication_range (float): the maximum range each robot can broadcast its position and map
    blocked_by_obstacles (bool): whether messages are blocked by obstacles

    Public Methods
    --------------
    add_robot(robot): adds a Robot to the team
    explore(world):  given the world the robot is exploring, iteratively explores the area
    """

    # Tunable parameter
    local_interaction_dist = 100.0
    local_interaction_path_length = 6

    def __init__(self, world_size, communication_range = float('inf'), blocked_by_obstacles = False):
        self._robots = {}
        self._communication_range = communication_range
        self._blocked_by_obstacles = blocked_by_obstacles
        self._initialize_map(world_size=world_size)


    # Private Methods
    def _initialize_map(self, world_size):
        """
        Initialized both the internal pixel and hex maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        """

        self._pixel_map = -np.ones(world_size)
        self._hex_map = convert_pixelmap_to_grid(pixel_map=self._pixel_map, size=AbstractRobot.hexagon_size)


    def _generate_message(self, robot_id, world):
        """
        Generates the message that a given robot with robot_id will receive

        Parameters
        ----------
        robot_id (str): the unique id of the robot to receive the message
        world (World): the world which contains the positions of every robot
        """

        message = {}
        robot_position = np.array(world.get_position(robot_id=robot_id))

        for robot in self._robots.values():
            if (robot.robot_id != robot_id):
                other_robot_position = np.array(world.get_position(robot.robot_id))
                distance = np.linalg.norm(robot_position - other_robot_position) * world.pixel_size

                if distance < self._communication_range:
                    if self._blocked_by_obstacles == False or world.clear_path_between_robots(robot1=robot.robot_id, robot2=robot_id):
                        message[robot.robot_id] = { 
                            'robot_position': other_robot_position,
                            'pixel_map': robot.pixel_map
                        }
        
        return message


    def _local_interaction(self, robot_states):
        """
        Checks if any two robots are too close to one another

        Parameters
        ----------
        robot_states (dict): a dictionary storing the RobotStates of each robot
    
        Returns
        -------
        is_local_interaction (bool): whether a local interaction occured
        """

        for robot in robot_states.keys():
            for second_robot in robot_states.keys():
                vector_between_robots = np.array(robot_states[robot].pixel_position) - np.array(robot_states[second_robot].pixel_position)
                dist_between_robots = np.linalg.norm(vector_between_robots)

                if robot != second_robot and dist_between_robots < 1.0:
                    return True
                elif robot != second_robot and dist_between_robots < self.local_interaction_dist:
                    robot_hex = self._hex_map.find_hex(desired_hex=self._hex_map.hex_at(point=robot_states[robot].pixel_position))
                    clear_path = check_distance_to_other_robot(hex_map=self._hex_map, robot_states=robot_states.values(), start_hex=robot_hex, max_hex_distance=self.local_interaction_path_length)
                    
                    if clear_path:
                        return True

        return False


    # Public Methods
    def add_robot(self, robot):
        """
        Adds a new robot to the team if it is not already in the team

        Parameters
        ----------
        robot (Robot): the robot to be added
        """

        if robot.robot_id not in self._robots:
            self._robots[robot.robot_id] = robot


    def explore(self, world):
        """
        Given the world the robot is exploring, iteratively explores the area with the whole team

        Parameters
        ----------
        world (World): a World object that the robot will explore
        """

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)

        for robot in self._robots.values():
            robot.complete_rotation(world=world)
            self._pixel_map = merge_map(hex_map=self._hex_map, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)

        self._hex_map.propagate_rewards()

        iteration = 0
        explored_per_iteration = []

        while self._hex_map.has_rewards() and iteration < 1000 and self._hex_map.percent_explored()/0.93 < 0.99:
            print(iteration)
            
            for robot in self._robots.values():
                message = self._generate_message(robot_id=robot.robot_id,  world=world)
                robot.communicate(message=message, iteration=iteration)

            t0 = time()
            for robot in self._robots.values():
                robot.explore_1_timestep(world=world, iteration=iteration)
                self._pixel_map = merge_map(hex_map=self._hex_map, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)
            print("Explore time", time()-t0)

            self._hex_map.propagate_rewards()

            with open('./plot_grids.txt', 'r') as reader:
                text = reader.read()
                
                if 'TRUE' in text:
                    plot_grid(grid=self._robots['robot_1'].hex_map, plot=ax1, robot_states=world.robot_states, mode='value')
                    plot_grid(grid=self._robots['robot_2'].hex_map, plot=ax2, robot_states=world.robot_states, mode='value')
                    plt.pause(0.05)
            
            grid_statistics =  [self._hex_map.percent_explored(), self._local_interaction(robot_states=world.robot_states)]
            explored_per_iteration.append(grid_statistics)

            print(grid_statistics)
            
            iteration += 1
        
        with open('./decentralized_exploration/results/two_robots_map_4/mdp_blocked.pkl', 'rb') as infile:
            all_results = pickle.load(infile)
        
        all_results.append(np.array(explored_per_iteration))

        with open('./decentralized_exploration/results/two_robots_map_4/mdp_blocked.pkl', 'wb') as outfile:
            pickle.dump(all_results, outfile, pickle.HIGHEST_PROTOCOL)
        
        plt.close('all')
