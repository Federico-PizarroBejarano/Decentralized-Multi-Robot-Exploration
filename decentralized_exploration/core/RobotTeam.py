import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

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
    robots (dict): a dictionary storing the RobotStates of each robot 
        using their robot_ids as keys
    communication_range (float): the maximum range each robot can broadcast its position and map
    blocked_by_obstacles (bool): whether messages are blocked by obstacles

    Public Methods
    --------------
    add_robot(robot): adds a Robot to the team
    explore(world):  given the world the robot is exploring, iteratively explores the area
    """

    # Tunable parameter
    local_interaction_dist = 100.0
    local_interaction_path_length = 8

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

        for robot in robot_states.values():
            for second_robot in robot_states.values():
                vector_between_robots = np.array(robot.pixel_position) - np.array(second_robot.pixel_position)
                dist_between_robots = np.linalg.norm(vector_between_robots)

                if robot != second_robot and dist_between_robots < self.local_interaction_dist:
                    robot_hex = self._hex_map.find_hex(desired_hex=self._hex_map.hex_at(point=robot.pixel_position))
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

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for robot in self._robots.values():
            robot.complete_rotation(world=world)
        
        iteration = 0
        explored_per_iteration = []

        while self._robots.values()[0].hex_map.has_rewards():
            print(iteration)
            for robot in self._robots.values():
                robot.communicate(message = self._generate_message(robot_id=robot.robot_id,  world=world), iteration=iteration)
            
            for robot in self._robots.values():
                robot.explore_1_timestep(world=world, iteration=iteration)
                self._pixel_map = merge_map(hex_map=self._hex_map, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)
                self._hex_map.propagate_rewards()

                hex_states = [h.state for h in self._hex_map.all_hexes.values()]
                grid_statistics =  [hex_states.count(-1), hex_states.count(0), hex_states.count(1), self._local_interaction(robot_states=world.robot_states)]
                print(grid_statistics)
                explored_per_iteration.append(grid_statistics)

                plot_grid(grid=self._hex_map, plot=ax, robot_states=world.robot_states, mode='reward')
                plt.pause(0.05)
            
            iteration += 1
        
        with open('./decentralized_exploration/results/greedy_mapmerger.pkl', 'rb') as infile:
            all_results = pickle.load(infile)
        
        all_results.append(np.array(explored_per_iteration))

        with open('./decentralized_exploration/results/greedy_mapmerger.pkl', 'wb') as outfile:
            pickle.dump(all_results, outfile, pickle.HIGHEST_PROTOCOL)
