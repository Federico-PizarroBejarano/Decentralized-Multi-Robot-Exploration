import numpy as np

from decentralized_exploration.core.robots.RobotMDP import RobotMDP
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import possible_actions, get_new_state, solve_MDP, compute_probability
from decentralized_exploration.helpers.grid import Cell, Grid, merge_map


class RobotMDP_Ind(RobotMDP):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotMDP_Ind, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private methods
    def _compute_DVF(self, current_cell, iteration, horizon):
        """
        Updates the repulsive value at each state. This is then used in _choose_next_pose to avoid other robots

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        horizon (int): how near a state is from the current state to be considered in the MDP
        """

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.cell_distance(self.grid.all_cells[robot['last_known_position']], current_cell) < self.horizon and robot_id != self.robot_id]

        DVF = {state : 0 for state in self._all_states}

        for robot in close_robots:
            if self._known_robots[robot]['last_updated'] != iteration:
                continue
            
            self._calculate_V(current_robot=robot, horizon=horizon)
            
            robot_cell = self.grid.all_cells[self._known_robots[robot]['last_known_position']]
            compute_probability(start_cell=robot_cell,
                                time_increment=0,
                                exploration_horizon=self.exploration_horizon,
                                grid=self.grid)

            for state in self._all_states:
                DVF[state] +=  self.weighing_factor * self.grid.all_cells[(state[0], state[1])].probability * self._known_robots[robot]['V'][state] + self.weighing_factor**3 * self.grid.all_cells[(state[0], state[1])].probability**2
        
        for state in self._all_states:
            DVF[state] = abs(DVF[state])

        return DVF
