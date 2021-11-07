import numpy as np
from heapdict import heapdict

from decentralized_exploration.core.constants import UNEXPLORED, UNOCCUPIED, Actions


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


def initialize_grid_for_search(start_cell, grid):
    for cell in grid.all_cells.values():
        cell.distance_from_start = float('inf')
        cell.visited = False
        cell.previous_cell = None
        
    start_cell.visited = True
    start_cell.distance_from_start = 0


def initialize_cell_queue(start_cell, grid, robot_states):
    cell_queue = heapdict() 

    other_robots = [robot_states[robot].pixel_position for robot in robot_states if robot_states[robot].pixel_position != start_cell.coord]

    neighbours = grid.cell_neighbours(center_cell=start_cell, radius=1)
    for neighbour in neighbours:
        if neighbour.state == 0: 
            neighbour.distance_from_start = start_cell.distance_from_start + 1
            neighbour.previous_cell = start_cell

            cell_queue[neighbour] = neighbour.distance_from_start

            # Do not explore the neighbours with a robot in them
            if neighbour.coord in other_robots:
                neighbour.visited = True
                cell_queue.popitem()

    return cell_queue 


def closest_frontier_cell(start_cell, grid, robot_states):
    '''
    Uses Dijkstra's algorithm to find the nearest free, frontier Cell.

    Parameters
    ----------
    start_cell (Cell): the starting Cell position
    grid (Grid): the Grid object representing the grid 

    Returns
    -------
    cell (tuple): the closest frontier Cell
    ''' 

    initialize_grid_for_search(start_cell, grid)
    cell_queue = initialize_cell_queue(start_cell, grid, robot_states, 'dijkstra')
      
    while(len(cell_queue) != 0):
        cell = cell_queue.popitem()[0]
        if cell.state == UNOCCUPIED:
            for neighbour in grid.cell_neighbours(cell):
                new_distance_from_start = cell.distance_from_start + 1
                if new_distance_from_start < neighbour.distance_from_start:
                    neighbour.previous_cell = cell
                    neighbour.distance_from_start = new_distance_from_start 
                    cell_queue[neighbour] = new_distance_from_start

                # Terminate when we found a frontier cell (a cell that neighbours an unxplored cell)
                if neighbour.state == UNEXPLORED:
                    return cell
    return None


def get_next_cell(start_cell, end_cell):
    next_cell = end_cell
    
    while next_cell.previous_cell != start_cell and next_cell.previous_cell != None:
        next_cell = next_cell.previous_cell

    return next_cell