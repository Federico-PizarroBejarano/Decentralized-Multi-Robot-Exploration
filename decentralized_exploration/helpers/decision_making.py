import numpy as np

from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.grid import Cell, Grid


def possible_actions(state, grid, robot_states):
    '''
    Given a state and the grid, returns all possible actions 

    Parameters
    ----------
    state (tuple): tuple of state (y, x)
    grid (Grid): a Grid representing the map
    robot_states (dict): a dictionary storing the RobotStates of each robot
    
    Returns
    -------
    poss_actions (list): a list of Actions (either up, down, left, right, up_left, up_right, down_left, down_right)
    '''
    other_robots = [robot_states[robot].pixel_position for robot in robot_states if robot_states[robot].pixel_position != state]

    poss_actions = []

    y, x = state
    
    if ((y+1, x) in grid.all_cells) and (grid.all_cells[(y+1, x)].state == 0) and ((y+1, x) not in other_robots):
        poss_actions.append(Actions.UP)
    if ((y-1, x) in grid.all_cells) and (grid.all_cells[(y-1, x)].state == 0) and ((y-1, x) not in other_robots):
        poss_actions.append(Actions.DOWN)
    if ((y, x+1) in grid.all_cells) and (grid.all_cells[(y, x+1)].state == 0) and ((y, x+1) not in other_robots):
        poss_actions.append(Actions.RIGHT)
    if ((y, x-1) in grid.all_cells) and (grid.all_cells[(y, x-1)].state == 0) and ((y, x-1) not in other_robots):
        poss_actions.append(Actions.LEFT)
    if ((y+1, x+1) in grid.all_cells) and (grid.all_cells[(y+1, x+1)].state == 0) and ((y+1, x+1)not in other_robots):
        poss_actions.append(Actions.UP_RIGHT)
    if ((y+1, x-1) in grid.all_cells) and (grid.all_cells[(y+1, x-1)].state == 0) and ((y+1, x-1) not in other_robots):
        poss_actions.append(Actions.UP_LEFT)
    if ((y-1, x+1) in grid.all_cells) and (grid.all_cells[(y-1, x+1)].state == 0) and ((y-1, x+1) not in other_robots):
        poss_actions.append(Actions.DOWN_RIGHT)
    if ((y-1, x-1) in grid.all_cells) and (grid.all_cells[(y-1, x-1)].state == 0) and ((y-1, x-1) not in other_robots):
        poss_actions.append(Actions.DOWN_LEFT)
    
    if poss_actions == []:
        poss_actions.append(Actions.STAY_STILL)
    
    return poss_actions


def get_new_state(state, action):
    '''
    Given a state and an action, computes the resulting state

    Parameters
    ----------
    state (tuple): tuple of state (y, x)
    action (Action): an Action (either up, down, left, right, up_left, up_right, down_left, down_right)

    Returns
    -------
    new_state (tuple): tuple of the new state (y, x)
    '''

    y, x = state
    
    if action == Actions.UP:
        new_state = (y+1, x)
    elif action == Actions.DOWN:
        new_state = (y-1, x)
    elif action == Actions.RIGHT:
        new_state = (y, x+1)
    elif action == Actions.LEFT:
        new_state = (y, x-1)
    elif action == Actions.UP_RIGHT:
        new_state = (y+1, x+1)
    elif action == Actions.UP_LEFT:
        new_state = (y+1, x-1)
    elif action == Actions.DOWN_RIGHT:
        new_state = (y-1, x+1)
    elif action == Actions.DOWN_LEFT:
        new_state = (y-1, x-1)
    elif action == Actions.STAY_STILL:
        new_state = state
    
    return new_state


def get_action(start_state, end_state):
    '''
    Given two neighbouring states, computes the necessary action

    Parameters
    ----------
    start_state (tuple): tuple of state (y, x)
    end_state (tuple): tuple of state (y, x)

    Returns
    -------
    action (Action): an Action (either up, down, left, right, up_left, up_right, down_left, down_right)
    '''

    if np.linalg.norm((start_state[0] - end_state[0], start_state[1] - end_state[1])) >= 2:
        print('States too distant!!')
        1/0
        
    y, x = start_state
    
    if end_state == (y+1, x):
        action = Actions.UP
    if end_state == (y-1, x):
        action = Actions.DOWN
    if end_state == (y, x+1):
        action = Actions.RIGHT
    if end_state == (y, x-1):
        action = Actions.LEFT
    if end_state == (y+1, x+1):
        action = Actions.UP_RIGHT
    if end_state == (y+1, x-1):
        action = Actions.UP_LEFT
    if end_state == (y-1, x+1):
        action = Actions.DOWN_RIGHT
    if end_state == (y-1, x-1):
        action = Actions.DOWN_LEFT
    if end_state == start_state:
        action = Actions.STAY_STILL
    
    return action


def solve_MDP(grid, V, rewards, noise, discount_factor, minimum_change, max_iterations, min_iterations, horizon, current_cell, DVF=None):
    '''
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
    '''

    policy = {}
    biggest_change = float('inf')
    iterations = 0

    all_states = V.keys()
    current_state = current_cell.coord

    if not DVF:
        DVF = {key:0 for key in V.keys()}

    while (biggest_change >= minimum_change or iterations < max(min_iterations, horizon) + 1) and (iterations < max(min_iterations, horizon, max_iterations) + 1):
        biggest_change = 0
        iterations += 1
        
        for state in all_states:
            if (state != current_state) and ( (grid.all_cells[state].state == 1) or (Grid.cell_distance(current_cell, Cell(state[0], state[1])) > horizon) ):
                continue

            old_value = V[state]
            new_value = -float('inf')
            
            for action in possible_actions(state=state, grid=grid, robot_states={}):
                next_state = get_new_state(state, action)

                value = rewards[state] + discount_factor * ( V[next_state] - DVF[next_state])
                # Keep best action so far
                if value > new_value:
                    new_value = value
                    policy[state] = action

            # Save best value            
            V[state] = new_value
            biggest_change = max(biggest_change, np.abs(old_value - V[state]))

    return policy


def max_value(V, current_state, poss_actions):
    '''
    Given a value function, the current state, and the possible actions, returns the optimal action

    Parameters
    ----------
    V (dict): a dictionary containing the value of each state
    current_state (tuple): the pixel coordinates of where the robot currently is
    poss_action (list): a list of Actions (either up, down, left, right, up_left, up_right, down_left, 
        down_right, or stay still)

    Returns
    -------
    best_action (Action): the optimal Action
    '''

    best_V = -float('inf')
    best_action = None

    for action in poss_actions:
        new_state = get_new_state(current_state, action)
        if V[new_state] > best_V:
            best_V = V[new_state]
            best_action = action
    
    return best_action


def compute_probability(start_cell, time_increment, exploration_horizon, grid):
    '''
    Calculates the probability that a robot has moved from start_cell to each other cell in time_increment. 
    Updates this information in the grid

    Parameters
    ----------
    start_cell (Cell): the Cell position that the robot was last known to occupy
    time_increment (int): the number of iterations of the algorithm since the robot was last contacted
    exploration_horizon (int): how far another cell can be and still be considered to be potentially explored
    grid (Grid): the Grid object representing the grid 
    '''

    for cell in grid.all_cells.values():
        cell.visited = False
        cell.distance_from_start = float('inf')
        
    start_cell.visited = True
    start_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=start_cell, radius=1)
    for neighbour in neighbours:
        if neighbour.state == 0:
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
            if num_possible_cells > 0:
                cell.probability = 1/(num_possible_cells * max(1.0, Grid.cell_distance(start_cell, cell)**0.5))
            else:
                cell.probability = 1.0


def calculate_utility(current_cell, grid, robot_in_sight, all_robots, alpha=1, beta=1):
    '''
    Calculates the utility metric of each reward cell.

    Parameters
    ----------
    current_cell (Cell): the cell where the robot currently is
    grid (Grid): a Grid object containing the map
    robot_states (list): a list storing the RobotStates of each robot in sight
    alpha (int): a tunable parameter
    beta (int): a tunable parameter

    Returns
    -------
    next_cell (Cell): the next cell the robot should go to to optimize utility
    '''

    calculate_coord(grid, robot_in_sight)
    calculate_dist(current_cell, grid, all_robots, alpha, beta)

    max_utility = -float('inf')
    max_cell = None

    for cell in grid.all_cells.values():
        if cell.reward > 0 and abs(cell.distance_from_start) != float('inf'):
            cell.utility = cell.norm_reward + cell.norm_dist - cell.coord_factor
            if cell.utility > max_utility:
                max_utility = cell.utility
                max_cell = cell
        else:
            cell.utility = -float('inf')
    
    if max_cell == None:
        return current_cell

    next_cell = max_cell
    while next_cell.previous_cell != current_cell and next_cell.previous_cell != None:
        next_cell = next_cell.previous_cell

    return next_cell


def calculate_dist(current_cell, grid, robot_states, alpha, beta):
    '''
    Given a grid and the current location of the robots, calculates the distance parameter
    to each reward cell to be used in the utility calculation

    Parameters
    ----------
    current_cell (Cell): the cell where the robot currently is
    grid (Grid): a Grid object containing the map
    robot_states (list): a list storing the RobotStates of each robot in sight
    alpha (int): a tunable parameter
    beta (int): a tunable parameter

    Returns
    -------
    next_cell (Cell): the next cell the robot should go to to optimize utility
    '''

    min_dist = float('inf')
    max_dist = 0
    max_reward = 0

    robot_cells = [grid.all_cells[robot_states[other_robot].pixel_position] for other_robot in robot_states.keys() if robot_states[other_robot].pixel_position != current_cell.coord]

    for cell in grid.all_cells.values():
        cell.visited = False
        cell.distance_from_start = float('inf')
        cell.previous_cell = None
        
    current_cell.visited = True
    current_cell.distance_from_start = 0

    if current_cell.reward > 0:
        min_dist = 0
        max_reward = current_cell.reward

    neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)

    for neighbour in neighbours:
        if neighbour.state == 0 and neighbour not in robot_cells:
            if neighbour.reward > 0:
                min_dist = min(min_dist, 1)
                max_dist = max(max_dist, 1)
                max_reward = max(max_reward, neighbour.reward)
            
            neighbour.distance_from_start = current_cell.distance_from_start + 1
            neighbour.previous_cell = current_cell

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.state == 0 and curr_cell.visited == False and curr_cell not in robot_cells:
            curr_cell.visited = True
            if curr_cell.reward > 0:
                min_dist = min(min_dist, curr_cell.distance_from_start)
                max_dist = max(max_dist, curr_cell.distance_from_start)
                max_reward = max(max_reward, curr_cell.reward)

            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    neighbour.previous_cell = curr_cell
                    cells_to_explore.append(neighbour)

    for cell in grid.all_cells.values():
        if max_dist == min_dist:
            cell.dist = 1.0
        else:
            cell.dist = float(cell.distance_from_start - min_dist)/float(max_dist - min_dist)
        
        cell.norm_dist = (cell.dist)**(alpha-1.0) * (1.0-cell.dist)**(beta-1.0)
        
        if max_reward == 0:
            cell.norm_reward = 0.0
        else:
            cell.norm_reward = float(cell.reward)/float(max_reward)


def calculate_coord(grid, robot_states):
    '''
    Given a grid and the locations of robots in sight, calculates the coordination factor
    to be used in the utility calculation.

    Parameters
    ----------
    grid (Grid): a Grid object containing the map
    robot_states (list): a list storing the RobotStates of each robot in sight
    '''

    if len(robot_states) == 0:
        return

    max_coord = 0

    for cell in grid.all_cells.values():
        cell.coord_factor = 0
    
    for robot in robot_states:
        max_dist = 0

        for cell in grid.all_cells.values():
            cell.visited = False
            cell.distance_from_start = float('inf')

        current_cell = grid.all_cells[robot.pixel_position]
        current_cell.visited = True
        current_cell.distance_from_start = 0

        neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)
        for neighbour in neighbours:
            if neighbour.state == 0:
                if neighbour.reward > 0:
                    max_dist = max(max_dist, 1)
                neighbour.distance_from_start = current_cell.distance_from_start + 1

        cells_to_explore = neighbours

        while(len(cells_to_explore) != 0):
            curr_cell = cells_to_explore.pop(0)
            if curr_cell.state == 0 and curr_cell.visited == False:
                curr_cell.visited = True
                if curr_cell.reward > 0:
                    max_dist = max(max_dist, curr_cell.distance_from_start)

                new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
                for neighbour in new_neighbours:
                    if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                        neighbour.distance_from_start = curr_cell.distance_from_start + 1
                        cells_to_explore.append(neighbour)
        
        for cell in grid.all_cells.values():
            if max_dist != 0:
                cell.coord_factor += float(max_dist - cell.distance_from_start)/float(max_dist)
            
            if cell.reward > 0:
                max_coord = max(cell.coord_factor, max_coord)

    for cell in grid.all_cells.values():
        if max_coord == 0:
            cell.coord_factor = 0
        else:
            cell.coord_factor = float(cell.coord_factor) / float(max_coord)


def closest_reward(current_cell, grid, robot_states):
    '''
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
    '''

    other_robots = [robot_states[robot].pixel_position for robot in robot_states if robot_states[robot].pixel_position != current_cell.coord]

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        cell.previous_cell = None
        
    current_cell.visited = True
    current_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)
    for neighbour in neighbours:
        if neighbour.state == 0:
            neighbour.distance_from_start = current_cell.distance_from_start + 1
            neighbour.previous_cell = current_cell

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.reward > 0 and curr_cell.coord not in other_robots:
            reward_cell = curr_cell
            max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell)
            while curr_cell.previous_cell != current_cell and curr_cell.previous_cell != None:
                curr_cell = curr_cell.previous_cell
                if Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell) > max_distance:
                    max_distance = Grid.cell_distance(start_cell=current_cell, end_cell=curr_cell)
            return curr_cell.coord, reward_cell, max_distance
        elif curr_cell.state == 0 and curr_cell.visited == False and curr_cell.coord not in other_robots:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.previous_cell = curr_cell
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    cells_to_explore.append(neighbour)
    
    return None, None, 0


def path_between_cells(current_cell, goal_cell, grid, robot_states):
    '''
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
    '''

    other_robots = [robot_states[robot].pixel_position for robot in robot_states if robot_states[robot].pixel_position != current_cell.coord]

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        cell.previous_cell = None
        
    current_cell.visited = True
    current_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=current_cell, radius=1)
    for neighbour in neighbours:
        if neighbour.state == 0 and neighbour not in robot_states:
            neighbour.distance_from_start = current_cell.distance_from_start + 1
            neighbour.previous_cell = current_cell

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.coord == goal_cell.coord:
            while curr_cell.previous_cell != current_cell and curr_cell.previous_cell != None:
                curr_cell = curr_cell.previous_cell
            return curr_cell.coord
        elif curr_cell.state == 0 and curr_cell.visited == False and curr_cell.coord not in other_robots:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.previous_cell = curr_cell
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    cells_to_explore.append(neighbour)
    
    return None
    

def check_distance_to_other_robot(grid, robot_states, start_cell, max_cell_distance):
    '''
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
    '''

    robot_cells = [robot.pixel_position for robot in robot_states]

    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        
    start_cell.visited = True
    start_cell.distance_from_start = 0

    neighbours = grid.cell_neighbours(center_cell=start_cell, radius=1)
    for neighbour in neighbours:
        if neighbour.state == 0:
            neighbour.distance_from_start = start_cell.distance_from_start + 1

    cells_to_explore = neighbours

    while(len(cells_to_explore) != 0):
        curr_cell = cells_to_explore.pop(0)
        if curr_cell.coord in robot_cells:
            return True
        elif curr_cell.state == 0 and curr_cell.visited == False:
            new_neighbours = grid.cell_neighbours(center_cell=curr_cell, radius=1)
            for neighbour in new_neighbours:
                if curr_cell.distance_from_start + 1 < neighbour.distance_from_start and neighbour.state == 0:
                    neighbour.distance_from_start = curr_cell.distance_from_start + 1
                    if neighbour.distance_from_start <= max_cell_distance:
                        cells_to_explore.append(neighbour)
    
    return False
               