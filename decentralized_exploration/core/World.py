import numpy as np

from decentralized_exploration.helpers.field_of_view import bresenham

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
    get_position(robot_id): returns the pixel position of the robot with the specified robot_id
    clear_path_between_robots(self, robot1, robot2): returns True if the path between two robots is 
        composed entirely of free pixels
    move_robot(robot_id, new_position): Updates the robot position of the robot with the specified robot_id
    """

    def __init__(self, world_map, pixel_size, robot_states):
        self._map = world_map
        self._pixel_size = pixel_size
        self._robot_states = robot_states

    @property
    def world_map(self):
        return self._map

    @property
    def pixel_size(self):
        return self._pixel_size
    
    @property
    def robot_states(self):
        return self._robot_states

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

        return self._robot_states[robot_id].pixel_position

    
    def clear_path_between_robots(self, robot1, robot2):
        """
        Returns True if the path between two robots is composed entirely of free pixels

        Parameters
        ----------
        robot1 (str): the robot_id of the first robot 
        robot2 (str): the robot_id of the second robot 

        Returns
        -------
        clear_path (bool): True is the line between the two robots is unoccupied, False otherwise
        """
        
        robot1_pos = self.get_position(robot1)
        robot1_pos = [int(coord) for coord in robot1_pos]
        robot2_pos = self.get_position(robot2)
        robot2_pos = [int(coord) for coord in robot2_pos]

        coords_of_line = bresenham(world_map=self._map, start=robot1_pos, end=robot2_pos)
        Y = [c[0] for c in coords_of_line]
        X = [c[1] for c in coords_of_line]
        points_in_line = self._map[Y, X]

        if np.any(points_in_line == 1):
            return False
        else:
            return True


    def move_robot(self, robot_id, new_position):
        """
        Updates the position for a robot with the given robot_id

        Parameters
        ----------
        robot_id (str): the id of the desired robot
        new_position (list): a 2-element list of integer pixel coordinates
        """

        self._robot_states[robot_id].pixel_position = new_position
