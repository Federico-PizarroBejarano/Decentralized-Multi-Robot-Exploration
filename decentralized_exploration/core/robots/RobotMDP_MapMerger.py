import numpy as np

from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state, solve_MDP, compute_probability
from decentralized_exploration.helpers.hex_grid import Hex, Grid, merge_map


class RobotMDP_MapMerger(RobotMDP):
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this
    minimum_change_repulsive (float): the repulsive MDP exits when the largest change in Value is less than this
    max_iterations (int): the maximum number of iterations before the MDP returns
    horizon (int): how near a state is from the current state to be considered in the MDP

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
    known_robots (dict): a dictionary containing the last known position and last time of contact
        with every other known robot in the team
    all_states (list)
    V (dict): a dictionary containing the value at each state, indexed by state
    repulsive_V (dict): a dictionary containing the repulsive value at each state, indexed by state

    Public Methods
    --------------
    explore_1_timestep(world): Explores the world for a single timestep/action. 
    """

    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotMDP_MapMerger, self).__init__(robot_id, range_finder, width, length, world_size)
 

    # Public Methods
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

        for robot_id in message:
            if robot_id not in self._known_robots:
                self._known_robots[robot_id] = {
                    'V': {state : self.hex_map.all_hexes[(state[0], state[1])].reward for state in self._all_states},
                    'repulsive_V': {state : 0 for state in self._all_states}
                }
            
            self._known_robots[robot_id]['last_updated'] = iteration
            self._known_robots[robot_id]['last_known_position'] = message[robot_id]['robot_position']

            self.__pixel_map = merge_map(hex_map=self.hex_map, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
        
        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
            'V': self._V,
            'repulsive_V': self._repulsive_V
        }
