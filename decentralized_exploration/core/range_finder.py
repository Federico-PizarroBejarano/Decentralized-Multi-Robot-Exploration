import numpy as np
from decentralized_exploration.helpers.field_of_view import field_of_view


class RangeFinder:
    """
    A class used to represent a range_finder of any kind

    Instance Attributes
    -------------------
    full_range (int): maximum distance in meters the RangeFinder can reliably detect distance
    frequency (int): the frequency the RangeFinder scans

    Public Methods
    --------------
    scan(world, new_orientation): object to scans the given world
    """

    def __init__(self, full_range, frequency):
        self.__full_range = full_range
        self.__frequency = frequency

    @property
    def full_range(self):
        return self.__full_range

    @property
    def frequency(self):
        return self.__frequency

    # Public Methods
    def scan(self, world, new_orientation, is_clockwise):
        """
        Scans the given world

        Parameters
        ----------
        world (World): a World object that the RangeFinder will scan
        new_orientation (int): an int 1-6 representing the new orientation of the robot
        is_clockwise (bool): True if the rotation is clockwise, False if counter_clockwise

        Returns
        -------
        all_free_points (list): a list of pixel coordinates representing free points
        all_occupied_points (list): a list of pixel coordinates representing occupied points
        """

        return field_of_view(world.world_map, world.robot_position, world.robot_orientation, new_orientation, is_clockwise)
