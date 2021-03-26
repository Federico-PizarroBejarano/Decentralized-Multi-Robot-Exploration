from decentralized_exploration.core.robots.RobotGreedy import RobotGreedy
from decentralized_exploration.helpers.hex_grid import merge_map


class RobotGreedy_MapMerger(RobotGreedy):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotGreedy_MapMerger, self).__init__(robot_id, range_finder, width, length, world_size)


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
            self.__pixel_map = merge_map(hex_map=self.hex_map, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
        
        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
        }
    