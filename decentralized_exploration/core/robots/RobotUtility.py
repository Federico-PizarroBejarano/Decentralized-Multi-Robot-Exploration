import numpy as np

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.helpers.decision_making import calculate_utility, possible_actions, get_action, get_new_state
from decentralized_exploration.core.constants import Actions, probability_of_failed_action
from decentralized_exploration.helpers.grid import merge_map


class RobotUtility(AbstractRobot):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotUtility, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private Methods
    def _choose_next_pose(self, current_position, iteration, robot_states):
        '''
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        robot_states (dict): a dictionary storing the RobotStates of each robot

        Returns
        -------
        next_state (tuple): tuple of q and r coordinates of the new position
        '''

        current_cell = self.grid.all_cells[current_position]

        robots_in_sight = [robot_id for robot_id in self._known_robots.keys() if self._known_robots[robot_id]['last_updated'] == iteration and robot_id != self.robot_id]
        robot_states_in_sight = [robot_states[robot_id] for robot_id in robot_states.keys() if robot_id in robots_in_sight]

        next_state = calculate_utility(current_cell=current_cell, grid=self.grid, robot_states=robot_states_in_sight, alpha=1, beta=2)


        if np.random.randint(100) > probability_of_failed_action:
            return next_state.coord
        else:
            actions = possible_actions(current_position, self.grid, robot_states) + [Actions.STAY_STILL]
            current_action = get_action(current_position, next_state.coord)
            actions.remove(current_action)
            next_action = np.random.choice(actions)

            return get_new_state(current_position, next_action)


    # Public Methods
    def communicate(self, message, iteration):
        '''
        Does nothing other than initialize the self._known_robots dictionary with itself.

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        iteration (int): the current iteration
        '''

        for robot_id in message:  
            if robot_id not in self._known_robots:
                self._known_robots[robot_id] = {}          
            self._known_robots[robot_id]['last_updated'] = iteration
            self._known_robots[robot_id]['last_known_position'] = message[robot_id]['robot_position']

            if message[robot_id]['pixel_map'] != []:
                self.__pixel_map = merge_map(grid=self.grid, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
                self.grid.propagate_rewards()

        self._known_robots[self.robot_id] = {
            'last_updated': iteration
        }
