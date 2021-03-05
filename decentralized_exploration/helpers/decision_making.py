from ..core.constants import Actions

def find_new_orientation(current_hex, current_orientation, next_hex):
    """
    Given the current hex and orientation, as well as the new hex, determines the new orientation of 
    the robot as well as whether it turns clockwise or counter-clockwise

    Parameters
    ----------
    current_hex (Hex): the Hex object the robot is currently at
    current_orientation (int): an int from 1-6 representing the current orientation
    next_hex (Hex): the hex object representing the next hex the robot will be or the hex the robot is 
        trying to look at

    Returns
    -------
    new_orientation (int): an int from 1-6 representing the new orientation
    is_clockwise (bool): True if the rotation is clockwise, False if counter-clockwise
    """

    curr_q, curr_r, curr_s = current_hex.q, current_hex.r, current_hex.s
    next_q, next_r, next_s = next_hex.q, next_hex.r, next_hex.s

    if curr_q == next_q:
        if next_r < curr_r:
            new_orientation = 1
        else:
            new_orientation = 4
    elif curr_r == next_r:
        if next_q < curr_q:
            new_orientation = 2
        else:
            new_orientation = 5
    elif curr_s == next_s:
        if next_q < curr_q:
            new_orientation = 3
        else:
            new_orientation = 6
    else:
        delta_q, delta_r, delta_s = next_q - curr_q, next_r - curr_r, next_s - curr_s
        if delta_q > 0 and delta_r > 0 and delta_s < 0:
            poss_orientations = [4, 5]
        elif delta_q > 0 and delta_r < 0 and delta_s < 0:
            poss_orientations = [5, 6]
        elif delta_q > 0 and delta_r < 0 and delta_s > 0:
            poss_orientations = [6, 1]
        elif delta_q < 0 and delta_r < 0 and delta_s > 0:
            poss_orientations = [1, 2]
        elif delta_q < 0 and delta_r > 0 and delta_s > 0:
            poss_orientations = [2, 3]
        elif delta_q < 0 and delta_r > 0 and delta_s < 0:
            poss_orientations = [3, 4]

        dist_1 = distance_between_orientations(start_orientation=current_orientation, end_orientation=poss_orientations[0])
        dist_2 = distance_between_orientations(start_orientation=current_orientation, end_orientation=poss_orientations[1])

        if dist_1 > dist_2:
            new_orientation = poss_orientations[0]
            intermediate_orientation = poss_orientations[1]
        else:
            new_orientation = poss_orientations[1]
            intermediate_orientation = poss_orientations[0]
        
        current_orientation = intermediate_orientation
    
    if (current_orientation < new_orientation and not (current_orientation == 1 and new_orientation == 6)) or (current_orientation == 6 and new_orientation == 1):
        is_clockwise = False
    else:
        is_clockwise = True
    
    return is_clockwise


def distance_between_orientations(start_orientation, end_orientation):
    """
    Given two orientations, determines the minimum number of 60 degree rotations necessary to get
    from one orientation to the other, called orientation distance

    Parameters
    ----------
    start_orientation (int): an int from 1-6 representing the starting orientation
    end_orientation (int): an int from 1-6 representing the ending orientation

    Returns
    -------
    orientation_distance (int): an int representing the orientation distance
    """

    one_way = (start_orientation - end_orientation) % 6
    another_way = (end_orientation - start_orientation) % 6

    orientation_distance = min(one_way, another_way)
    return orientation_distance


def possible_actions(state, hex_map):
    poss_actions = [Actions.COUNTER_CLOCKWISE, Actions.CLOCKWISE]

    orientation = state[2]
    
    if orientation == 1:
        next_position = (state[0], state[1] - 1)
    elif orientation == 2:
        next_position = (state[0] - 1, state[1])
    elif orientation == 3:
        next_position = (state[0] - 1, state[1] + 1)
    elif orientation == 4:
        next_position = (state[0], state[1] + 1)
    elif orientation == 5:
        next_position = (state[0] + 1, state[1])
    elif orientation == 6:
        next_position = (state[0] + 1, state[1] - 1)
    
    if (next_position in hex_map.all_hexes) and (hex_map.all_hexes[next_position].state == 0):
        poss_actions.append(Actions.FORWARD)
    
    return poss_actions


def get_new_state(state, action):
    if action == Actions.FORWARD:
        orientation = state[2]
        
        if orientation == 1:
            next_state = (state[0], state[1] - 1, orientation)
        elif orientation == 2:
            next_state = (state[0] - 1, state[1], orientation)
        elif orientation == 3:
            next_state = (state[0] - 1, state[1] + 1, orientation)
        elif orientation == 4:
            next_state = (state[0], state[1] + 1, orientation)
        elif orientation == 5:
            next_state = (state[0] + 1, state[1], orientation)
        elif orientation == 6:
            next_state = (state[0] + 1, state[1] - 1, orientation)
    elif action == Actions.COUNTER_CLOCKWISE:
        new_orientation = state[2] + 1 if state[2] + 1 <= 6 else 1 
        next_state = (state[0], state[1], new_orientation)
    elif action == Actions.CLOCKWISE:
        new_orientation = state[2] - 1 if (state[2] - 1 >= 1) else 6 
        next_state = (state[0], state[1], new_orientation)
    
    return next_state
