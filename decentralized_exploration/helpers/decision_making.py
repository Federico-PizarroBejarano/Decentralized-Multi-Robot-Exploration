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
    is_clockwise (bool): True if the rotation is clockwise, False if counter-clockwise
    new_orientation (int): an int from 1-6 representing the new orientation
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
    
    return is_clockwise, new_orientation


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


def solve_MDP(hex_map, V, rewards, noise, discount_factor, minimum_change, max_iterations, min_iterations, horizon, current_hex, DVF=None):
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
    min_iterations (int): the minimum number of iterations before the MDP returns
    horizon (int): how near a state is from the current state to be considered in the MD
    current_hex (Hex): the hex of where the robot currently is
    DVF (dict): the distributed value function to be subtracted from the value function

    Returns
    -------
    policy (dict): a dictionary containing the optimal action to perform at each state, indexed by state
    """

    policy = {}
    biggest_change = float('inf')
    iterations = 0

    all_states = V.keys()
    current_state = (current_hex.q, current_hex.r)

    if not DVF:
        DVF = {key:0 for key in V.keys()}

    while (biggest_change >= minimum_change or iterations < max(min_iterations, horizon) + 1) and (iterations < max(min_iterations, horizon, max_iterations) + 1):
        biggest_change = 0
        iterations += 1
        
        for state in all_states:
            if ((state[0], state[1]) != current_state) and ( (hex_map.all_hexes[(state[0], state[1])].state == 1) or (Grid.hex_distance(current_hex, Hex(state[0], state[1])) > horizon) ):
                continue

            old_value = V[state]
            new_value = -float('inf')
            
            for action in possible_actions(state, hex_map):
                next_state = get_new_state(state, action)

                # Choose a random action to do with probability noise
                random_state = next_state
                if noise > 0.0:
                    random_action = np.random.choice([rand_act for rand_act in possible_actions(state, hex_map) if rand_act != action])
                    random_state = get_new_state(state, random_action)

                value = rewards[(state[0], state[1])] + discount_factor * ( ((1 - noise)* V[next_state] + (noise * V[random_state])) - DVF[next_state])
                
                # Keep best action so far
                if value > new_value:
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
            
            try:
                if pixel_map[v0[0], v0[1]] == 0 and pixel_map[v1[0], v1[1]] == 0:
                    voronoi_path += bresenham(pixel_map, v0, v1)
            except Exception as e:
                print(e)
    return np.array(voronoi_path)


def compute_probability(start_hex, time_increment, exploration_horizon, hex_map):
    """
    Calculates the probability that a robot has moved from start_hex to each other hex in time_increment. 
    Updates this information in the hex_map

    Parameters
    ----------
    start_hex (Hex): the Hex position that the robot was last known to occupy
    time_increment (int): the number of iterations of the algorithm since the robot was last contacted
    exploration_horizon (int): how far another hex can be and still be considered to be potentially explored
    hex_map (Grid): the Grid object representing the hex_map 
    """

    for hexagon in hex_map.all_hexes.values():
        hexagon.visited = False
        hexagon.distance_from_start = float('inf')
        
    start_hex.visited = True
    start_hex.distance_from_start = 0

    neighbours = hex_map.hex_neighbours(center_hex=start_hex, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = start_hex.distance_from_start + 1

    hexes_to_explore = neighbours
    num_possible_hexes = 0

    while(len(hexes_to_explore) != 0):
        curr_hex = hexes_to_explore.pop(0)
        if curr_hex.state == 0 and curr_hex.visited == False and curr_hex.distance_from_start < exploration_horizon + time_increment/2:
            curr_hex.visited = True
            num_possible_hexes += 1

            new_neighbours = hex_map.hex_neighbours(center_hex=curr_hex, radius=1)
            for neighbour in new_neighbours:
                if curr_hex.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_hex.distance_from_start + 1
                    hexes_to_explore.append(neighbour)
                    
    for hexagon in hex_map.all_hexes.values():
        if hexagon.visited:
            hexagon.probability = 1/(num_possible_hexes * max(1.0, Grid.hex_distance(start_hex, hexagon)**0.5))


def closest_reward(current_hex, hex_map):
    """
    Uses breadth first search to find the nearest free Hex with a reward.

    Parameters
    ----------
    current_hex (Hex): the current Hex position
    hex_map (Grid): the Grid object representing the hex_map 

    Returns
    -------
    next_state (tuple): the next state the robot should go to as a tuple of 
        q and r coordinates of the new position
    reward_hex (Hex): the closest Hex that has a reward
    max_distance (int): the largest distance between the current_hex and a hex 
        in the path to the reward_hex. Used to calculate the necessary horizon
    """

    for hexagon in hex_map.all_hexes.values():
        hexagon.distance_from_start = float('inf')
        hexagon.visited = False
        hexagon.previous_hex = None
        
    current_hex.visited = True
    current_hex.distance_from_start = 0

    neighbours = hex_map.hex_neighbours(center_hex=current_hex, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = current_hex.distance_from_start + 1
        neighbour.previous_hex = current_hex

    hexes_to_explore = neighbours

    while(len(hexes_to_explore) != 0):
        curr_hex = hexes_to_explore.pop(0)
        if curr_hex.reward > 0:
            reward_hex = curr_hex
            max_distance = Grid.hex_distance(start_hex=current_hex, end_hex=curr_hex)
            while curr_hex.previous_hex != current_hex and curr_hex.previous_hex != None:
                curr_hex = curr_hex.previous_hex
                if Grid.hex_distance(start_hex=current_hex, end_hex=curr_hex) > max_distance:
                    max_distance = Grid.hex_distance(start_hex=current_hex, end_hex=curr_hex)
            return (curr_hex.q, curr_hex.r), reward_hex, max_distance
        elif curr_hex.state == 0 and curr_hex.visited == False:
            new_neighbours = hex_map.hex_neighbours(center_hex=curr_hex, radius=1)
            for neighbour in new_neighbours:
                if curr_hex.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.previous_hex = curr_hex
                    neighbour.distance_from_start = curr_hex.distance_from_start + 1
                    hexes_to_explore.append(neighbour)
    
    return None, None, 0
    

def check_distance_to_other_robot(hex_map, robot_states, start_hex, max_hex_distance):
    """
    Checks if there is another robot within the max_hex_distance of the start_hex

    Parameters
    ----------
    hex_map (Grid): the Grid object representing the hex_map 
    robot_states (dict): a dictionary storing the RobotStates of each robot
    start_hex (Hex): the Hex which currently has a robot
    max_hex_distance (int): the maximum path length between the two robots

    Returns
    -------
    is_local_interaction (bool): whether a local interaction occured
    """

    robot_hexes = [hex_map.hex_at(point=robot.pixel_position) for robot in robot_states]
    robot_hex_positions = [(robot_hex.q, robot_hex.r) for robot_hex in robot_hexes]   

    for hexagon in hex_map.all_hexes.values():
        hexagon.distance_from_start = float('inf')
        hexagon.visited = False
        
    start_hex.visited = True
    start_hex.distance_from_start = 0

    neighbours = hex_map.hex_neighbours(center_hex=start_hex, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = start_hex.distance_from_start + 1

    hexes_to_explore = neighbours

    while(len(hexes_to_explore) != 0):
        curr_hex = hexes_to_explore.pop(0)
        if (curr_hex.q, curr_hex.r) in robot_hex_positions:
            return True
        elif curr_hex.state == 0 and curr_hex.visited == False:
            new_neighbours = hex_map.hex_neighbours(center_hex=curr_hex, radius=1)
            for neighbour in new_neighbours:
                if curr_hex.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_hex.distance_from_start + 1
                    if neighbour.distance_from_start <= max_hex_distance:
                        hexes_to_explore.append(neighbour)
    
    return False
               