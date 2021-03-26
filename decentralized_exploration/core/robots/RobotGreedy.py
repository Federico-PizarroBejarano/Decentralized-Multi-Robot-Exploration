from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, get_new_state, closest_reward
from decentralized_exploration.helpers.hex_grid import Hex


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
        
        if on_reward_hex:
            next_hex = self.hex_map.find_closest_unknown(center_hex=current_hex)
            is_clockwise = find_new_orientation(current_hex=current_hex, current_orientation=current_orientation, next_hex=next_hex)
            action = Actions.CLOCKWISE if is_clockwise else Actions.COUNTER_CLOCKWISE
            next_state = get_new_state(current_state, action)
        else:
            next_position = closest_reward(current_hex, self.hex_map)
            state_forward = get_new_state(current_state, Actions.FORWARD)

            if state_forward[0] == next_position[0] and state_forward[1] == next_position[1]:
                next_state = state_forward
            else:
                next_hex = Hex(next_position[0], next_position[1])
                is_clockwise = find_new_orientation(current_hex=current_hex, current_orientation=current_orientation, next_hex=next_hex)
                action = Actions.CLOCKWISE if is_clockwise else Actions.COUNTER_CLOCKWISE
                next_state = get_new_state(current_state, action)

        return next_state
    