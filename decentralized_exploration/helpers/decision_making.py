import numpy as np

from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.grid import Cell, Grid
from decentralized_exploration.helpers.field_of_view import bresenham


def possible_actions(state, grid):
    """
    Given a state and the grid, returns all possible actions 

    Parameters
    ----------
    state (tuple): tuple of state (y, x)
    grid (Grid): a Grid representing the map

    Returns
    -------
    poss_actions (list): a list of Actions (either up, down, left, right, up_left, up_right, down_left, down_right)
    """

    poss_actions = []

    y, x = state[0], state[1]
    
    if ((y+1, x) in grid.all_cells) and (grid.all_cells[(y+1, x)].state == 0):
        poss_actions.append(Actions.UP)
    if ((y-1, x) in grid.all_cells) and (grid.all_cells[(y-1, x)].state == 0):
        poss_actions.append(Actions.DOWN)
    if ((y, x+1) in grid.all_cells) and (grid.all_cells[(y, x+1)].state == 0):
        poss_actions.append(Actions.RIGHT)
    if ((y, x-1) in grid.all_cells) and (grid.all_cells[(y, x-1)].state == 0):
        poss_actions.append(Actions.LEFT)
    if ((y+1, x+1) in grid.all_cells) and (grid.all_cells[(y+1, x+1)].state == 0):
        poss_actions.append(Actions.UP_RIGHT)
    if ((y+1, x-1) in grid.all_cells) and (grid.all_cells[(y+1, x-1)].state == 0):
        poss_actions.append(Actions.UP_LEFT)
    if ((y-1, x+1) in grid.all_cells) and (grid.all_cells[(y-1, x+1)].state == 0):
        poss_actions.append(Actions.DOWN_RIGHT)
    if ((y-1, x-1) in grid.all_cells) and (grid.all_cells[(y-1, x-1)].state == 0):
        poss_actions.append(Actions.DOWN_LEFT)
    
    return poss_actions


def get_new_state(state, action):
    """
    Given a state and an action, computes the resulting state

    Parameters
    ----------
    state (tuple): tuple of state (y, x)
    action (Action): an Action (either up, down, left, right, up_left, up_right, down_left, down_right)

    Returns
    -------
    new_state (tuple): tuple of the new state (y, x)
    """
    y, x = state[0], state[1]
    
    if action == Actions.UP:
        new_state = (y+1, x)
    if action == Actions.DOWN:
        new_state = (y-1, x)
    if action == Actions.RIGHT:
        new_state = (y, x+1)
    if action == Actions.LEFT:
        new_state = (y, x-1)
    if action == Actions.UP_RIGHT:
        new_state = (y+1, x+1)
    if action == Actions.UP_LEFT:
        new_state = (y+1, x-1)
    if action == Actions.DOWN_RIGHT:
        new_state = (y-1, x+1)
    if action == Actions.DOWN_LEFT:
        new_state = (y-1, x-1)
    
    return new_state


def solve_MDP(grid, V, rewards, noise, discount_factor, minimum_change, max_iterations, min_iterations, horizon, current_cell, DVF=None):
    """
    Solves an MDP given the states, rewards, transition function, and actions. 

    Parameters
    ----------
    grid (Grid): a Grid object containing the map. 
    V (dict): a dictionary containing the initial guess for the value of each state
    rewards (dict): a dictionary of the reward at each positional state
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this
    max_iterations (int): the maximum number of iterations before the MDP returns
    min_iterations (int): the minimum number of iterations before the MDP returns
    horizon (int): how near a state is from the current state to be considered in the MD
    current_cell (Cell): the cell of where the robot currently is
    DVF (dict): the distributed value function to be subtracted from the value function

    Returns
    -------
    policy (dict): a dictionary containing the optimal action to perform at each state, indexed by state
    """

    policy = {}
    biggest_change = float('inf')
    iterations = 0

    all_states = V.keys()
    current_state = (current_cell.q, current_cell.r)

    if not DVF:
        DVF = {key:0 for key in V.keys()}

    while (biggest_change >= minimum_change or iterations < max(min_iterations, horizon) + 1) and (iterations < max(min_iterations, horizon, max_iterations) + 1):
        biggest_change = 0
        iterations += 1
        
        for state in all_states:
            if ((state[0], state[1]) != current_state) and ( (grid.all_cells[(state[0], state[1])].state == 1) or (Grid.cell_distance(current_cell, Cell(state[0], state[1])) > horizon) ):
                continue

            old_value = V[state]
            new_value = -float('inf')
            
            for action in possible_actions(state, grid):
                next_state = get_new_state(state, action)

                # Choose a random action to do with probability noise
                random_state = next_state
                if noise > 0.0:
                    random_action = np.random.choice([rand_act for rand_act in possible_actions(state, grid) if rand_act != action])
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


def compute_probability(start_cell, time_increment, exploration_horizon, grid):
    """
    Calculates the probability that a robot has moved from start_cell to each other cell in time_increment. 
    Updates this information in the grid

    Parameters
    ----------
    start_cell (Cell): the Cell position that the robot was last known to occupy
    time_increment (int): the number of iterations of the algorithm since the robot was last contacted
    exploration_horizon (int): how far another cell can be and still be considered to be potentially explored
    grid (Grid): the Grid object representing the grid 
    """

    for cell in grid.all_cells.values():
        cell.visited = False
        cell.distance_from_start = float('inf')
        
    start_cell.visited = True
    start_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=start_cell, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = start_cell.distance_from_start + 1

    cells_to_explore = neighbours
    num_possible_cells = 0

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.state == 0 and curr_cell.visited == False and curr_cell.distance_from_start < exploration_horizon + time_increment/2:
            curr_cell.visited = True
            num_possible_cells += 1

            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    cells_to_explore.append(neighbour)
                    
    for cell in grid.all_cells.values():
        if cell.visited:
            cell.probability = 1/(num_possible_cells * max(1.0, Grid.cell_distance(start_cell, cell)**0.5))


def closest_reward(current_cell, grid):
    """
    Uses breadth first search to find the nearest free Cell with a reward.

    Parameters
    ----------
    current_cell (Cell): the current Cell position
    grid (Grid): the Grid object representing the grid 

    Returns
    -------
    next_state (tuple): the next state the robot should go to as a tuple of 
        q and r coordinates of the new position
    reward_cell (Cell): the closest Cell that has a reward
    max_distance (int): the largest distance between the current_cell and a cell 
        in the path to the reward_cell. Used to calculate the necessary horizon
    """

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        cell.previous_cell = None
        
    current_cell.visited = True
    current_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = current_cell.distance_from_start + 1
        neighbour.previous_cell = current_cell

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.reward > 0:
            reward_cell = curr_cell
            max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell)
            while curr_cell.previous_cell != current_cell and curr_cell.previous_cell != None:
                curr_cell = curr_cell.previous_cell
                if Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell) > max_distance:
                    max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell)
            return (curr_cell.q, curr_cell.r), reward_cell, max_distance
        elif curr_cell.state == 0 and curr_cell.visited == False:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.previous_cell = curr_cell
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    cells_to_explore.append(neighbour)
    
    return None, None, 0


def path_between_cells(current_cell, goal_cell, grid):
    """
    Uses breadth first search to find the nearest free Cell with a reward.

    Parameters
    ----------
    current_cell (Cell): the current Cell position
    goal_cell (Cell): the goal Cell position
    grid (Grid): the Grid object representing the grid 

    Returns
    -------
    next_state (tuple): the next state the robot should go to as a tuple of 
        y and x coordinates of the new position
    """

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        cell.previous_cell = None
        
    current_cell.visited = True
    current_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = current_cell.distance_from_start + 1
        neighbour.previous_cell = current_cell

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if (curr_cell.y, curr_cell.x) == (goal_cell.y, goal_cell.x):
            final_cell = curr_cell
            max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=final_cell)
            while curr_cell.previous_cell != current_cell and curr_cell.previous_cell != None:
                curr_cell = curr_cell.previous_cell
                if Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell) > max_distance:
                    max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell)
            return (curr_cell.q, curr_cell.r)
        elif curr_cell.state == 0 and curr_cell.visited == False:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.previous_cell = curr_cell
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    cells_to_explore.append(neighbour)
    
    return None
    

def check_distance_to_other_robot(grid, robot_states, start_cell, max_cell_distance):
    """
    Checks if there is another robot within the max_cell_distance of the start_cell

    Parameters
    ----------
    grid (Grid): the Grid object representing the grid 
    robot_states (dict): a dictionary storing the RobotStates of each robot
    start_cell (Cell): the Cell which currently has a robot
    max_cell_distance (int): the maximum path length between the two robots

    Returns
    -------
    is_local_interaction (bool): whether a local interaction occured
    """

    robot_cells = [robot.pixel_position for robot in robot_states]

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        
    start_cell.visited = True
    start_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=start_cell, radius=1)
    for neighbour in neighbours:
        neighbour.distance_from_start = start_cell.distance_from_start + 1

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if (curr_cell.y, curr_cell.x) in robot_cells:
            return True
        elif curr_cell.state == 0 and curr_cell.visited == False:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    if neighbour.distance_from_start <= max_cell_distance:
                        cells_to_explore.append(neighbour)
    
    return False
               