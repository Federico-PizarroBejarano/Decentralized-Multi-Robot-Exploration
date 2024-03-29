from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, get_new_state, closest_reward
from decentralized_exploration.helpers.hex_grid import Hex, merge_map


class RobotGreedy(AbstractRobot):
    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotGreedy, self).__init__(robot_id, range_finder, width, length, world_size)


    # Private Methods
    def _choose_next_pose(self, current_position, current_orientation, iteration):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        current_orientation (int): int representing current orientation of robot
        iteration (int): the current iteration of the algorithm

        Returns
        -------
        next_state (tuple): tuple of q and r coordinates of the new position, with orientation at the end
        """

        current_hex_pos = self.hex_map.hex_at(point=current_position)
        current_hex = self.hex_map.find_hex(desired_hex=current_hex_pos)
        current_state = (current_hex.q, current_hex.r, current_orientation)
        
        # Checking if on reward hexagon
        on_reward_hex = current_hex.reward > 0
        
        if on_reward_hex and not self._escaping_dead_reward['escaping_dead_reward']:          
            next_hex = self.hex_map.find_closest_unknown(center_hex=current_hex)
            is_clockwise, new_orientation = find_new_orientation(current_hex=current_hex, current_orientation=current_orientation, next_hex=next_hex)

            if new_orientation == current_orientation:
                if next_hex.state == 0:
                    action = Actions.FORWARD
                    next_state = get_new_state(current_state, action)
                    return next_state
                else:
                    self._escaping_dead_reward['escaping_dead_reward'] = True 
            else:
                if self._escaping_dead_reward['was_just_on_reward'] == True and new_orientation == self._escaping_dead_reward['previous_orientation']:
                    self._escaping_dead_reward['escaping_dead_reward'] = True 
                else:
                    self._escaping_dead_reward['was_just_on_reward'] = True
                    self._escaping_dead_reward['previous_orientation'] = current_orientation
                    action = Actions.CLOCKWISE if is_clockwise else Actions.COUNTER_CLOCKWISE
                    next_state = get_new_state(current_state, action)
                    return next_state
        
        self._escaping_dead_reward['was_just_on_reward'] = False
        next_position = closest_reward(current_hex, self.hex_map)[0]

        # All rewards have been found
        if next_position == None:
            return current_state

        next_hex = Hex(next_position[0], next_position[1])
        is_clockwise, new_orientation = find_new_orientation(current_hex=current_hex, current_orientation=current_orientation, next_hex=next_hex)

        if new_orientation == current_orientation:
            action = Actions.FORWARD
            self._escaping_dead_reward['escaping_dead_reward'] = False
        else:
            action = Actions.CLOCKWISE if is_clockwise else Actions.COUNTER_CLOCKWISE
        next_state = get_new_state(current_state, action)

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
            self.__pixel_map = merge_map(hex_map=self.hex_map, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
            self.hex_map.propagate_rewards()

        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
        }