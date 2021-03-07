import numpy as np


class World:
    """
    A class used to represent a single robot

    Instance Attributes
    -------------------
    world_map (numpy.ndarry): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    pixel_size (float): the height a pixel represents in the map in meters
    robot_states (dict): a dictionary storing the RobotStates of each robot 
        using their robot_ids as keys

    Public Methods
    --------------
    move_robot(new_position, new_orientation): Updates robot position and orientation
    """

    def __init__(self, world_map, pixel_size, robot_states):
        self.__map = world_map
        self.__pixel_size = pixel_size
        self.__robot_states = robot_states

    @property
    def world_map(self):
        return self.__map

    @property
    def pixel_size(self):
        return self.__pixel_size
    
    @property
    def robot_states(self):
        return self.__robot_states

    # Public Methods
    def get_position(self, robot_id):
        """
        Returns the pixel position of the robot with the given robot_id

        Parameters
        ----------
        robot_id (str): the id of the desired robot

        Returns
        -------
        position (list): a 2-element list of integer pixel coordinates
        """

        return self.__robot_states[robot_id].pixel_position
    

    def get_orientation(self, robot_id):
        """
        Returns the orientation of the robot with the given robot_id

        Parameters
        ----------
        robot_id (str): the id of the desired robot

        Returns
        -------
        orientation (int): an int representing the orientation of the robot
        """

        return self.__robot_states[robot_id].orientation


    def move_robot(self, robot_id, new_position, new_orientation):
        """
        Updates the position and orientation for a robot with the given robot_id

        Parameters
        ----------
        robot_id (str): the id of the desired robot
        new_position (list): a 2-element list of integer pixel coordinates
        new_orientation (int): an integer representing the orientation of the robot
        """

        self.__robot_states[robot_id].pixel_position = new_position
        self.__robot_states[robot_id].orientation = new_orientation
