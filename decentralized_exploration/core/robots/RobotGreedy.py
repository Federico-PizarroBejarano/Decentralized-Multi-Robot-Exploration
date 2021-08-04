from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import get_new_state, closest_reward, path_between_cells
from decentralized_exploration.helpers.grid import Cell, merge_map


class RobotGreedy(AbstractRobot):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotGreedy, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private Methods
    def _choose_next_pose(self, current_position, iteration):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm

        Returns
        -------
        next_state (tuple): tuple of q and r coordinates of the new position
        """

        current_cell = self.grid.all_cells[current_position]
        
        goal_position = closest_reward(current_cell, self.grid)[0]

        # All rewards have been found
        if goal_position == None:
            return current_position

        goal_cell = Cell(goal_position[0], goal_position[1])

        next_state = path_between_cells(current_cell, goal_cell, self.grid)
        
        return next_state
    

    # Public Methods
    def communicate(self, message, iteration):
        """
        Does nothing other than initialize the self._known_robots dictionary with itself.

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        iteration (int): the current iteration
        """

        for robot_id in message:
            self.__pixel_map = merge_map(grid=self.grid, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
            self.grid.propagate_rewards()

        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
        }