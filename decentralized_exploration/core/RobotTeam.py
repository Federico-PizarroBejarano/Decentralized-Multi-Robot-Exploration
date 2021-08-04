import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.helpers.grid import convert_pixelmap_to_grid, merge_map
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
        Initialized both the internal pixel and cell maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        """

        self._pixel_map = -np.ones(world_size)
        self._grid = convert_pixelmap_to_grid(pixel_map=self._pixel_map)


    def _generate_message(self, robot_id, world):
        """
        Generates the message that a given robot with robot_id will receive

        Parameters
        ----------
        robot_id (str): the unique id of the robot to receive the message
        world (World): the world which contains the positions of every robot

        Returns
        -------
        message (dict): a dictionary of robot positions and pixel maps for each robot, 
            indexed by their robot_ids
        """

        message = {}
        robot_position = np.array(world.get_position(robot_id=robot_id))

        for robot in self._robots.values():
            if (robot.robot_id != robot_id):
                other_robot_position = np.array(world.get_position(robot.robot_id))
                distance = max(abs(robot_position[1]-other_robot_position[1]), abs(robot_position[0]-other_robot_position[0]))

                if distance <= self._communication_range:
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
                    robot_cell = self._grid.all_cells[robot_states[robot].pixel_position]
                    clear_path = check_distance_to_other_robot(grid=self._grid, robot_states=robot_states.values(), start_cell=robot_cell, max_cell_distance=self.local_interaction_path_length)
                    
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
        ax2 = fig1.add_subplot(111)

        fig2 = plt.figure(2)
        ax1 = fig2.add_subplot(111)
    
        plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode='reward')
        plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode='reward')
        
        plt.pause(0.05)

        for robot in self._robots.values():
            robot.scan_environment(world=world)
            self._pixel_map = merge_map(grid=self._grid, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)
            # plot_grid(grid=self._grid, plot=ax1, robot_states=world.robot_states, mode='reward')
            plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode='reward')
            plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode='reward')
            plt.pause(0.05)

        self._grid.propagate_rewards()

        iteration = 0
        # explored_per_iteration = []

        while self._grid.has_rewards() and iteration < 1000:
            print("Iteration #", iteration)
            
            for robot in self._robots.values():
                message = self._generate_message(robot_id=robot.robot_id,  world=world)
                robot.communicate(message=message, iteration=iteration)

            for robot in self._robots.values():
                robot.explore_1_timestep(world=world, iteration=iteration)
                self._pixel_map = merge_map(grid=self._grid, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)

            self._grid.propagate_rewards()

            # plot_grid(grid=self._grid, plot=ax1, robot_states=world.robot_states, mode='reward')
            plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode='reward')
            plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode='reward')
            plt.pause(0.5)
            
            # grid_statistics =  [self._grid.percent_explored(), self._local_interaction(robot_states=world.robot_states), world.get_position('robot_1'), world.get_position('robot_2')]
            # explored_per_iteration.append(grid_statistics)
            
            iteration += 1

        # with open('./decentralized_exploration/results/trajectories/mdp_no_comm.pkl', 'wb') as outfile:
        #     pickle.dump(explored_per_iteration, outfile, pickle.HIGHEST_PROTOCOL)
        
        plt.close('all')
