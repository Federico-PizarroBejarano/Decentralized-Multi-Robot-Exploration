import numpy as np

from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state, solve_MDP, compute_probability
from decentralized_exploration.helpers.hex_grid import Hex, Grid, merge_map


class RobotMDP_Ind(RobotMDP):
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
        super(RobotMDP_Ind, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private methods
    def _compute_DVF(self, current_hex, iteration):
        """
        Updates the repulsive value at each state. This is then used in _choose_next_pose to avoid other robots

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        """

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.hex_distance(self.hex_map.hex_at(robot['last_known_position']), current_hex) < self.horizon and robot_id != self.robot_id]

        DVF = {state : 0 for state in self._all_states}

        for robot in close_robots:
            if self._known_robots[robot]['last_updated'] != iteration:
                continue
            
            self._calculate_V(current_robot=robot)
            
            robot_hex = self.hex_map.find_hex(self.hex_map.hex_at(point=self._known_robots[robot]['last_known_position']))
            compute_probability(start_hex=robot_hex,
                                time_increment=iteration - self._known_robots[robot]['last_updated'],
                                exploration_horizon=self.exploration_horizon,
                                hex_map=self.hex_map)

            for state in self._all_states:
                DVF[state] +=  self.weighing_factor * self.hex_map.all_hexes[(state[0], state[1])].probability * self._known_robots[robot]['V'][state]
        
        for state in self._all_states:
            DVF[state] = abs(DVF[state])

        return DVF
