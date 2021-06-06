import numpy as np

from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state, solve_MDP, compute_probability
from decentralized_exploration.helpers.hex_grid import Hex, Grid, merge_map


class RobotMDP_Ind(RobotMDP):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotMDP_Ind, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private methods
    def _compute_DVF(self, current_hex, iteration, horizon):
        """
        Updates the repulsive value at each state. This is then used in _choose_next_pose to avoid other robots

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        horizon (int): how near a state is from the current state to be considered in the MDP
        """

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.hex_distance(self.hex_map.hex_at(robot['last_known_position']), current_hex) < self.horizon and robot_id != self.robot_id]

        DVF = {state : 0 for state in self._all_states}

        for robot in close_robots:
            if self._known_robots[robot]['last_updated'] != iteration:
                continue
            
            self._calculate_V(current_robot=robot, horizon=horizon)
            
            robot_hex = self.hex_map.find_hex(self.hex_map.hex_at(point=self._known_robots[robot]['last_known_position']))
            compute_probability(start_hex=robot_hex,
                                time_increment=0,
                                exploration_horizon=self.exploration_horizon,
                                hex_map=self.hex_map)

            for state in self._all_states:
                DVF[state] +=  self.weighing_factor * self.hex_map.all_hexes[(state[0], state[1])].probability * self._known_robots[robot]['V'][state] + self.weighing_factor**3 * self.hex_map.all_hexes[(state[0], state[1])].probability**2
        
        for state in self._all_states:
            DVF[state] = abs(DVF[state])

        return DVF
