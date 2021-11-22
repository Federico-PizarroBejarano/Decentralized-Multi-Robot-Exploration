import numpy as np
from abc import ABCMeta, abstractmethod
from decentralized_exploration.core.constants import UNEXPLORED

from decentralized_exploration.core.environment.grid import convert_pixelmap_to_grid


class AbstractRobot:
    '''
    A class used to represent a single robot

    Instance Attributes
    -------------------
    robot_id (str): the unique id of this robot
    range_finder (RangeFinder): a RangeFinder object representing the sensor
    width (float): the width of the robot in meters
    length (float): the length of the robot in meters
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    grid (Grid): A Grid object holding the grid layer

    Public Methods
    --------------
    explore_1_timestep(world): Explores the world for a single timestep/action. 
    '''

    __metaclass__ = ABCMeta

    
    def __init__(self, robot_id, range_finder, width, length, world_size):
        self._robot_id = robot_id
        self._range_finder = range_finder
        self._width = width
        self._length = length
        self._escaping_dead_reward = {
            'was_just_on_reward': False, 
            'escaping_dead_reward': False
        }

        self._known_robots = { robot_id: {}}

        self._initialize_map(world_size=world_size)

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def size(self):
        return [self._width, self._length]

    @property
    def pixel_map(self):
        return self._pixel_map

    @property
    def grid(self):
        return self._grid


    # Private methods
    def _initialize_map(self, world_size):
        '''
        Initialized both the internal pixel and grid given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        '''

        self._pixel_map = -np.ones(world_size)
        self._grid = convert_pixelmap_to_grid(pixel_map=self.pixel_map)


    def _update_map(self, occupied_points, free_points):
        '''
        Updates both the internal pixel and grid given lists of occupied and free pixels

        Parameters
        ----------
        occupied_points (list of [x, y] points): list of occupied points
        free_points (list of [x, y] points): list of free points
        '''

        occupied_points = [p for p in occupied_points if self.pixel_map[p[0], p[1]] == -1]
        free_points = [p for p in free_points if self.pixel_map[p[0], p[1]] == -1]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.pixel_map[occ_rows, occ_cols] = 1
        self.pixel_map[free_rows, free_cols] = 0

        for occ_point in occupied_points:
            found_cell = self.grid.all_cells[occ_point]
            found_cell.update_cell(state=1)

        for free_point in free_points:
            found_cell = self.grid.all_cells[free_point]
            found_cell.update_cell(state=0)


    def _update_frontier_after_scan(self, free_points):
        self.grid.cleanup_frontier()

        for free_point in free_points:
            free_cell = self.grid.all_cells[free_point]
            neighbours = self.grid.cell_neighbours(center_cell=free_cell, radius=1)

            for neighbour in neighbours:
                if neighbour.state == UNEXPLORED:
                    self.grid.frontier.add(free_point)
                    break
    

    @abstractmethod
    def _choose_next_pose(self, current_position, iteration, robot_states):
        '''
        Given the current pos, decides on the next best position for the robot. 
        Overriden in each sub-class. 

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        robot_states (dict): a dictionary storing the RobotStates of each robot

        Returns
        -------
        next_state (tuple): tuple of y and x coordinates of the new position
        '''

        pass


    # Public Methods
    @abstractmethod
    def communicate(self, message, iteration):
        '''
        Communicates with the other robots in the team. Overridden in each sub-class. 

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        iteration (int): the current iteration
        '''

        pass


    def scan_environment(self, world):
        occupied_points, free_points = self._range_finder.scan(world=world, position=world.get_position(self.robot_id))

        self._update_map(occupied_points=occupied_points, free_points=free_points)
        self._update_frontier_after_scan(free_points)


    def explore_1_timestep(self, world, iteration):
        '''
        Given the world the robot is exploring, explores the area for 1 timestep/action

        Parameters
        ----------
        world (World): a World object that the robot will explore
        iteration (int): the current iteration of the algorithm
        '''

        self._known_robots[self.robot_id]['last_known_position'] = world.get_position(self.robot_id)
        
        new_position = self._choose_next_pose(current_position=world.get_position(self.robot_id), iteration=iteration, robot_states=world.robot_states)
        
        self.scan_environment(world=world)

        world.move_robot(robot_id=self.robot_id, new_position=new_position)
