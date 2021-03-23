import numpy as np
from scipy.spatial import Voronoi # pylint: disable-msg=E0611

from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.hex_grid import Hex, Grid
from decentralized_exploration.helpers.field_of_view import bresenham

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
    """
    Given a state and the hex_map, returns all possible actions 

    Parameters
    ----------
    state (tuple): tuple of state (q, r, orientation)
    hex_map (Grid): a Grid representing the map

    Returns
    -------
    poss_actions (list): a list of Actions (either forward, clockwise, or counter_clockwise)
    """

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
    """
    Given a state and an action, computes the resulting state

    Parameters
    ----------
    state (tuple): tuple of state (q, r, orientation)
    action (Action): an Action (either forward, clockwise, or counter_clockwise)

    Returns
    -------
    new_state (tuple): tuple of the new state (q, r, orientation)
    """

    if action == Actions.FORWARD:
        orientation = state[2]
        
        if orientation == 1:
            new_state = (state[0], state[1] - 1, orientation)
        elif orientation == 2:
            new_state = (state[0] - 1, state[1], orientation)
        elif orientation == 3:
            new_state = (state[0] - 1, state[1] + 1, orientation)
        elif orientation == 4:
            new_state = (state[0], state[1] + 1, orientation)
        elif orientation == 5:
            new_state = (state[0] + 1, state[1], orientation)
        elif orientation == 6:
            new_state = (state[0] + 1, state[1] - 1, orientation)
    elif action == Actions.COUNTER_CLOCKWISE:
        new_orientation = state[2] + 1 if state[2] + 1 <= 6 else 1 
        new_state = (state[0], state[1], new_orientation)
    elif action == Actions.CLOCKWISE:
        new_orientation = state[2] - 1 if (state[2] - 1 >= 1) else 6 
        new_state = (state[0], state[1], new_orientation)
    
    return new_state


def solve_MDP(hex_map, V, rewards, noise, discount_factor, minimum_change, max_iterations, horizon, current_hex, DVF=None):
    """
    Solves an MDP given the states, rewards, transition function, and actions. 

    Parameters
    ----------
    hex_map (Grid): a Grid object containing the map. 
    V (dict): a dictionary containing the initial guess for the value of each state
    rewards (dict): a dictionary of the reward at each positional state
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this
    max_iterations (int): the maximum number of iterations before the MDP returns
    horizon (int): how near a state is from the current state to be considered in the MD
    current_hex (Hex): the hex of where the robot currently is

    Returns
    -------
    policy (dict): a dictionary containing the optimal action to perform at each state, indexed by state
    """

    policy = {}
    biggest_change = float('inf')
    iterations = 0

    all_states = V.keys()

    if not DVF:
        DVF = {key:0 for key in V.keys()}

    while (biggest_change >= minimum_change) and (iterations < max_iterations):
        biggest_change = 0
        iterations += 1
        
        for state in all_states:
            if (hex_map.all_hexes[(state[0], state[1])].state != 0) or (Grid.hex_distance(current_hex, Hex(state[0], state[1])) > horizon):
                continue

            old_value = V[state]
            new_value = -float('inf')
            
            for action in possible_actions(state, hex_map):
                next_state = get_new_state(state, action)

                # Choose a random action to do with probability self.noise
                random_action = np.random.choice([rand_act for rand_act in possible_actions(state, hex_map) if rand_act != action])
                random_state = get_new_state(state, random_action)

                value = rewards[(state[0], state[1])] + discount_factor * ( ((1 - noise)* V[next_state] + (noise * V[random_state])) - DVF[next_state])
                
                # Keep best action so far
                if value >= new_value:
                    new_value = value
                    policy[state] = action

            # Save best value                         
            V[state] = new_value
            biggest_change = max(biggest_change, np.abs(old_value - V[state]))

    return policy


def voronoi_paths(pixel_map):
    """
    Given a map, finds the voronoi paths through the free space

    Parameters
    ----------
    pixel_map (numpy.ndarry): numpy array of pixels representing the map.

    Returns
    -------
    voronoi_path (numpy.ndarry): a numpy array of every point that is along a voronoi path
    """

    obstacle_points = np.argwhere(pixel_map != 0)
    
    vor = Voronoi(np.array(obstacle_points))

    voronoi_path = []

    for vpair in vor.ridge_vertices:
        if vpair[0] >= 0 and vpair[1] >= 0:
            v0 = vor.vertices[vpair[0]]
            v1 = vor.vertices[vpair[1]]

            v0 = [int(round(coord)) for coord in v0]
            v1 = [int(round(coord)) for coord in v1]
            if pixel_map[v0[0], v0[1]] == 0 and pixel_map[v1[0], v1[1]] == 0:
                voronoi_path += bresenham(pixel_map, v0, v1)
    
    return np.array(voronoi_path)


def compute_probability(start_hex, time_increment, exploration_horizon, hex_map):
    """
    Calculates the probability that a robot has moved from start_hex to new_hex in time_increment. 

    Parameters
    ----------
    start_hex (Hex): the Hex position that the robot was last known to occupy
    time_increment (int): the number of iterations of the algorithm since the robot was last contacted
    exploration_horizon (int): how far another hex can be and still be considered to be potentially explored
    hex_map (Grid): the Grid object representing the hex_map 

    Returns
    -------
    probability (float): the probability (from 0 - 1) that the robot is in the new state. 
    """

    for hexagon in hex_map.all_hexes.values():
        hexagon.probability = 0.0
        hexagon.probability_steps = float('inf')
        
    start_hex.probability = 1.0
    start_hex.probability_steps = 0

    neighbours = hex_map.hex_neighbours(center_hex=start_hex, radius=1)
    for neighbour in neighbours:
        neighbour.probability_steps = start_hex.probability_steps + 1

    hexes_to_explore = neighbours
    num_possible_hexes = 0

    while(len(hexes_to_explore) != 0):
        curr_hex = hexes_to_explore.pop(-1)
        if curr_hex.state == 0 and curr_hex.probability == 0 and curr_hex.probability_steps < exploration_horizon + time_increment/2:
            curr_hex.probability = 1.0
            num_possible_hexes += 1

            new_neighbours = hex_map.hex_neighbours(center_hex=curr_hex, radius=1)
            for neighbour in new_neighbours:
                neighbour.probability_steps = min(curr_hex.probability_steps + 1, neighbour.probability_steps)
            
            hexes_to_explore += new_neighbours
                    
    for hexagon in hex_map.all_hexes.values():
        if hexagon.probability > 0:
            hexagon.probability /= num_possible_hexes * max(1, Grid.hex_distance(start_hex, hexagon)**0.5)
                    