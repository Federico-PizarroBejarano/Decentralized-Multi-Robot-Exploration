import numpy as np
import matplotlib.pyplot as plt
import time

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.helpers.grid import convert_pixelmap_to_grid, merge_map
from decentralized_exploration.helpers.decision_making import check_distance_to_other_robot
from decentralized_exploration.helpers.plotting import plot_grid
from decentralized_exploration.core.constants import probability_of_failed_communication

class RobotTeam:
    '''
    A class used to represent a team of robots
    
    Class Attributes
    ----------------
    local_interaction_dist (float): the maximum pixel distance between robots to be considered
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
    '''

    # Tunable parameter
    local_interaction_dist = 4

    def __init__(self, world_size, communication_range = float('inf'), blocked_by_obstacles = False):
        self._robots = {}
        self._communication_range = communication_range
        self._blocked_by_obstacles = blocked_by_obstacles
        self._initialize_map(world_size=world_size)


    # Private Methods
    def _initialize_map(self, world_size):
        '''
        Initialized both the internal pixel and cell maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        '''

        self._pixel_map = -np.ones(world_size)
        self._grid = convert_pixelmap_to_grid(pixel_map=self._pixel_map)


    def _generate_message(self, robot_id, world):
        '''
        Generates the message that a given robot with robot_id will receive

        Parameters
        ----------
        robot_id (str): the unique id of the robot to receive the message
        world (World): the world which contains the positions of every robot

        Returns
        -------
        message (dict): a dictionary of robot positions and pixel maps for each robot, 
            indexed by their robot_ids
        '''

        message = {}
        robot_position = world.get_position(robot_id=robot_id)

        for robot in self._robots.values():
            if (robot.robot_id != robot_id):
                other_robot_position = world.get_position(robot.robot_id)
                distance = max(abs(robot_position[1]-other_robot_position[1]), abs(robot_position[0]-other_robot_position[0]))

                if distance <= self._communication_range:
                    if self._blocked_by_obstacles == False or world.clear_path_between_robots(robot1=robot.robot_id, robot2=robot_id):
                        if np.random.randint(100) > probability_of_failed_communication :
                            message[robot.robot_id] = { 
                                'robot_position': other_robot_position,
                                'pixel_map': robot.pixel_map
                            }
                        else:
                            message[robot.robot_id] = { 
                                'robot_position': other_robot_position,
                                'pixel_map': []
                            }
        
        return message


    def _local_interaction(self, robot_states, world):
        '''
        Checks if any two robots are too close to one another

        Parameters
        ----------
        robot_states (dict): a dictionary storing the RobotStates of each robot
        world (World): the world which contains the positions of every robot
    
        Returns
        -------
        is_local_interaction (bool): whether a local interaction occured
        '''

        local_interactions = []

        for robot in robot_states.keys():
            for second_robot in robot_states.keys():
                robot_position = robot_states[robot].pixel_position
                other_robot_position = robot_states[second_robot].pixel_position
                dist_between_robots = max(abs(robot_position[1]-other_robot_position[1]), abs(robot_position[0]-other_robot_position[0]))

                if robot != second_robot and dist_between_robots < self.local_interaction_dist:
                    if self._blocked_by_obstacles == False or world.clear_path_between_robots(robot1=robot, robot2=second_robot):                 
                        local_interactions.append((int(robot[-1]), int(second_robot[-1])))

        return local_interactions


    # Public Methods
    def add_robot(self, robot):
        '''
        Adds a new robot to the team if it is not already in the team

        Parameters
        ----------
        robot (Robot): the robot to be added
        '''

        if robot.robot_id not in self._robots:
            self._robots[robot.robot_id] = robot


    def explore(self, world):
        '''
        Given the world the robot is exploring, iteratively explores the area with the whole team

        Parameters
        ----------
        world (World): a World object that the robot will explore
        '''

        # fig1 = plt.figure(1)
        # ax1 = fig1.add_subplot(111)

        # fig2 = plt.figure(2)
        # ax2 = fig2.add_subplot(111)

        # fig3 = plt.figure(3)
        # ax3 = fig3.add_subplot(111)

        # mode = 'value'
    
        # plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode=mode)
        # plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode=mode)
        # plot_grid(grid=self._robots['robot_3'].grid, plot=ax3, robot_states=world.robot_states, mode=mode)
        
        # plt.pause(0.05)

        start_time = time.time()

        for robot in self._robots.values():
            robot.scan_environment(world=world)
            self._pixel_map = merge_map(grid=self._grid, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)
            # plot_grid(grid=self._grid, plot=ax1, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_3'].grid, plot=ax3, robot_states=world.robot_states, mode=mode)
            # plt.pause(0.05)

        self._grid.propagate_rewards()

        iteration = 0
        explored_per_iteration = []
        distances_travelled = [0, 0, 0]
        last_positions = [(10000, 10000), (10000, 10000), (10000, 10000)]

        while self._grid.has_rewards():
            if iteration >= 150:
                print('TAKING TOO LONG')
                1/0
            print('Iteration #', iteration, '  % explored: ', self._grid.percent_explored())
            
            for robot in self._robots.values():
                message = self._generate_message(robot_id=robot.robot_id,  world=world)
                robot.communicate(message=message, iteration=iteration)

            for robot in self._robots.values():
                # print(robot.robot_id)
                last_positions[int(robot.robot_id[-1])-1] = world.get_position(robot_id=robot.robot_id)
                robot.explore_1_timestep(world=world, iteration=iteration)
                self._pixel_map = merge_map(grid=self._grid, pixel_map=self._pixel_map, pixel_map_to_merge=robot.pixel_map)
                distances_travelled[int(robot.robot_id[-1])-1] += np.linalg.norm(np.array(last_positions[int(robot.robot_id[-1])-1]) - np.array(world.get_position(robot_id=robot.robot_id)))

            self._grid.propagate_rewards()

            # plot_grid(grid=self._grid, plot=ax1, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_1'].grid, plot=ax1, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_2'].grid, plot=ax2, robot_states=world.robot_states, mode=mode)
            # plot_grid(grid=self._robots['robot_3'].grid, plot=ax3, robot_states=world.robot_states, mode=mode)
            # plt.pause(0.5)
            
            grid_statistics =  [self._grid.percent_explored(), self._local_interaction(robot_states=world.robot_states, world=world), distances_travelled, world.get_position('robot_1'), world.get_position('robot_2'), world.get_position('robot_3'), time.time()-start_time]
            explored_per_iteration.append(grid_statistics)
            
            iteration += 1
        
        # plt.close('all')

        return explored_per_iteration
