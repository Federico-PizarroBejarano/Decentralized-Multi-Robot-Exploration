import numpy as np

class World:
    """
    A class used to represent a single robot

    Instance Attributes
    ----------
    world_map (numpy.ndarry): numpy array of pixels representing the map. 
        0  == free
        1  == occupied
    pixel_size (float): the height a pixel represents in the map in meters
    robot (decentralized_exploration.core.robot.Robot): A Robot object that will explore the world
    robot_pos (array): a 2-element array of integer pixel coordinates
    robot_orientation (int): an integer representing the orientation of the robot as facing one side of a hex (1-6)

    Public Methods
    -------
    move_robot(new_position, new_orientation): Updates robot position and orientation
    """

    def __init__(self, world_map, pixel_size, robot, robot_position, robot_orientation):
        self.__map = world_map
        self.__pixel_size = pixel_size
        self.__robot = robot
        self.__robot_pos = robot_position
        self.__robot_orientation = robot_orientation
    
    @property
    def world_map(self):
        return self.__map

    @property
    def robot_position(self):
        return self.__robot_pos
    
    @property
    def robot_orientation(self):
        return self.__robot_orientation
    

    def move_robot(self, new_position, new_orientation):
        """
        Updates the robot position and orientation

        Parameters
        ----------
        new_position (array): a 2-element array of integer pixel coordinates
        new_orientation (int): an integer representing the orientation of the robot as facing one side of a hex (1-6)
        """

        self.__robot_pos = new_position
        self.__robot_orientation = new_orientation 