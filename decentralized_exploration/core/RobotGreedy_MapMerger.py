from decentralized_exploration.core.RobotGreedy import RobotGreedy


class RobotGreedy_MapMerger(RobotGreedy):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        RobotGreedy.__init__(self, robot_id, range_finder, width, length, world_size)
        self._known_robots = { robot_id: {}}  

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
            self._merge_map(other_map=message[robot_id]['pixel_map'])
        
        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
        }
    