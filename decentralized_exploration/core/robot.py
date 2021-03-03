import numpy as np
import matplotlib.pyplot as plt

from decentralized_exploration.helpers.decision_making import find_new_orientation
from decentralized_exploration.helpers.hex_grid import convert_pixelmap_to_grid
from decentralized_exploration.helpers.plotting import plot_map, plot_grid


class Robot:
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter

    Instance Attributes
    -------------------
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
    explore(world): Starts the process of the robot exploring the world. Returns fully explored pixel map
    """
    # Tunable Parameter
    hexagon_size = 4

    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.__initialize_map(world_size=world_size)

    @property
    def size(self):
        return [self.__width, self.__length]

    @property
    def pixel_map(self):
        return self.__pixel_map

    @property
    def hex_map(self):
        return self.__hex_map

    # Private methods

    def __initialize_map(self, world_size):
        """
        Initialized both the internal pixel and hex maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        """

        self.__pixel_map = -np.ones(world_size)
        self.__hex_map = convert_pixelmap_to_grid(pixel_map=self.__pixel_map, size=self.hexagon_size)

    def __update_map(self, occupied_points, free_points):
        """
        Updates both the internal pixel and hex maps given arrays of occupied and free pixels

        Parameters
        ----------
        occupied_points (array of [x, y] points): array of occupied points
        free_points (array of [x, y] points): array of free points
        """

        occupied_points = [p for p in occupied_points if self.__pixel_map[p[0], p[1]] == -1]
        free_points = [p for p in free_points if self.__pixel_map[p[0], p[1]] == -1]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.__pixel_map[occ_rows, occ_cols] = 1
        self.__pixel_map[free_rows, free_cols] = 0

        for occ_point in occupied_points:
            desired_hex = self.__hex_map.hex_at(point=occ_point)
            found_hex = self.__hex_map.find_hex(desired_hex=desired_hex)
            self.__hex_map.update_hex(hex_to_update=found_hex, dOccupied=1, dUnknown=-1)

        for free_point in free_points:
            desired_hex = self.__hex_map.hex_at(point=free_point)
            found_hex = self.__hex_map.find_hex(desired_hex=desired_hex)
            self.__hex_map.update_hex(hex_to_update=found_hex, dFree=1, dUnknown=-1)
        
        self.__hex_map.propagate_rewards()

    def __choose_next_pose(self, current_pos):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_pos (tuple): tuple of integer pixel coordinates

        Returns
        -------
        new_pos (tuple): tuple of integer pixel coordinates of the new position
        new_orientation (int): an int representing the new orientation
        is_clockwise (bool): a bool representing whether the rotation is clockwise
        """

        # TBD

        return (0, 0), 1, True

    # Public Methods
    def explore(self, world):
        """
        Given the world the robot is exploring, iteratively explores the area

        Parameters
        ----------
        world (World): a World object that the robot will explore

        Returns
        -------
        pixel_map (numpy.ndarry): numpy array of pixels representing the fully explored map. 
        """

        fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(111)

        # Do a 360 scan
        world.move_robot(new_position=world.robot_position, new_orientation=6)
        for orientation in range(1, 6+1):
            occupied_points, free_points = self.__range_finder.scan(world=world, new_orientation=orientation, is_clockwise=False)
            self.__update_map(occupied_points=occupied_points, free_points=free_points)

            world.move_robot(new_position=world.robot_position, new_orientation=orientation)
            # plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
            plot_grid(grid=self.__hex_map, plot=ax2, robot_pos=world.robot_position, robot_orientation=world.robot_orientation)
            plt.pause(0.05)

        while self.__hex_map.has_unexplored():
            new_position, new_orientation, is_clockwise = self.__choose_next_pose(current_pos=world.robot_position)

            occupied_points, free_points = self.__range_finder.scan(world=world, new_orientation=new_orientation, is_clockwise=is_clockwise)

            self.__update_map(occupied_points=occupied_points, free_points=free_points)
            world.move_robot(new_position=new_position, new_orientation=new_orientation)

            # plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
            plot_grid(grid=self.__hex_map, plot=ax2, robot_pos=world.robot_position, robot_orientation=world.robot_orientation)
            plt.pause(0.05)

        return self.__pixel_map
