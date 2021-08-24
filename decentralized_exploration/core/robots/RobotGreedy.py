import numpy as np

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions, probability_of_failed_action
from decentralized_exploration.helpers.decision_making import get_new_state, closest_reward, path_between_cells, possible_actions, get_action
from decentralized_exploration.helpers.grid import Cell, merge_map


class RobotGreedy(AbstractRobot):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotGreedy, self).__init__(robot_id, range_finder, width, length, world_size)


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
        
        goal_position = closest_reward(current_cell=current_cell, grid=self.grid, robot_states=robot_states)[0]

        # All rewards have been found
        if goal_position == None:
            return current_position

        goal_cell = Cell(goal_position[0], goal_position[1])

        next_state = path_between_cells(current_cell=current_cell, goal_cell=goal_cell, grid=self.grid, robot_states=robot_states)
        
        if np.random.randint(100) > probability_of_failed_action:
            return next_state
        else:
            actions = possible_actions(current_position, self.grid, robot_states) + [Actions.STAY_STILL]
            current_action = get_action(current_position, next_state)
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
            if message[robot_id]['pixel_map'] != []:
                self.__pixel_map = merge_map(grid=self.grid, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
                self.grid.propagate_rewards()

        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
        }