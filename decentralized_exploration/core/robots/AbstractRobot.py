import numpy as np
from abc import ABCMeta, abstractmethod

from decentralized_exploration.helpers.hex_grid import Hex, convert_pixelmap_to_grid


class AbstractRobot:
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter

    Instance Attributes
    -------------------
    robot_id (str): the unique id of this robot
    range_finder (RangeFinder): a RangeFinder object representing the sensor
    width (float) : the width of the robot in meters
    length (float) : the length of the robot in meters
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    hex_map (Grid): A Grid object holding the hex layer

    Public Methods
    --------------
    explore_1_timestep(world): Explores the world for a single timestep/action. 
    """

    __metaclass__ = ABCMeta

    # Tunable Parameters
    hexagon_size = 9
    
    def __init__(self, robot_id, range_finder, width, length, world_size):
        self._robot_id = robot_id
        self._range_finder = range_finder
        self._width = width
        self._length = length
        self._escaping_dead_reward = False

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
    def hex_map(self):
        return self._hex_map


    # Private methods
    def _initialize_map(self, world_size):
        """
        Initialized both the internal pixel and hex maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        """

        self._pixel_map = -np.ones(world_size)
        self._hex_map = convert_pixelmap_to_grid(pixel_map=self.pixel_map, size=self.hexagon_size)


    def _update_map(self, occupied_points, free_points):
        """
        Updates both the internal pixel and hex maps given lists of occupied and free pixels

        Parameters
        ----------
        occupied_points (list of [x, y] points): list of occupied points
        free_points (list of [x, y] points): list of free points
        """

        occupied_points = [p for p in occupied_points if self.pixel_map[p[0], p[1]] == -1]
        free_points = [p for p in free_points if self.pixel_map[p[0], p[1]] == -1]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.pixel_map[occ_rows, occ_cols] = 1
        self.pixel_map[free_rows, free_cols] = 0

        for occ_point in occupied_points:
            desired_hex = self.hex_map.hex_at(point=occ_point)
            found_hex = self.hex_map.find_hex(desired_hex=desired_hex)
            found_hex.update_hex(dOccupied=1, dUnknown=-1)

        for free_point in free_points:
            desired_hex = self.hex_map.hex_at(point=free_point)
            found_hex = self.hex_map.find_hex(desired_hex=desired_hex)
            found_hex.update_hex(dFree=1, dUnknown=-1)
        
        self.hex_map.propagate_rewards()


    @abstractmethod
    def _choose_next_pose(self, current_position, current_orientation, iteration):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        current_orientation (int): int representing current orientation of robot
        iteration (int): the current iteration of the algorithm

        Returns
        -------
        next_state (tuple): tuple of q and r coordinates of the new position, with orientation at the end
        """

        pass


    # Public Methods
    def complete_rotation(self, world):
        """
        Rotates the robot completely to scan the area around it

        Parameters
        ----------
        world (World): a World object that the robot will explore
        """

        starting_orientation = world.get_orientation(self.robot_id)
        next_orientation = starting_orientation + 1 if (starting_orientation + 1 <= 6) else 1
        count = 0

        while count < 6:
            occupied_points, free_points = self._range_finder.scan(world=world, position=world.get_position(self.robot_id), old_orientation=world.get_orientation(self.robot_id), new_orientation=next_orientation, is_clockwise=False)
            self._update_map(occupied_points=occupied_points, free_points=free_points)

            world.move_robot(robot_id=self.robot_id, new_position=world.get_position(self.robot_id), new_orientation=next_orientation)
            next_orientation = next_orientation + 1 if (next_orientation + 1 <= 6) else 1

            count += 1

    def communicate(self, message, iteration):
        """
        Communicates with the other robots in the team. Receives a message and updates the 
        last known position and last updated time of every robot that transmitted a message. 
        Additionally, merges in all their pixel maps.

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        iteration (int): the current iteration
        """

        pass


    def explore_1_timestep(self, world, iteration):
        """
        Given the world the robot is exploring, explores the area for 1 timestep/action

        Parameters
        ----------
        world (World): a World object that the robot will explore
        iteration (int): the current iteration of the algorithm
        """

        self._known_robots[self.robot_id]['last_known_position'] = world.get_position(self.robot_id)

        new_state = self._choose_next_pose(current_position=world.get_position(self.robot_id), current_orientation=world.get_orientation(self.robot_id), iteration=iteration)
        new_position = self.hex_map.hex_center(Hex(new_state[0], new_state[1]))
        new_position = [int(coord) for coord in new_position]
        new_orientation = new_state[2]

        occupied_points, free_points = self._range_finder.scan(world=world, position=world.get_position(self.robot_id), old_orientation=world.get_orientation(self.robot_id), new_orientation=new_orientation)

        self._update_map(occupied_points=occupied_points, free_points=free_points)
        world.move_robot(robot_id=self.robot_id, new_position=new_position, new_orientation=new_orientation)
