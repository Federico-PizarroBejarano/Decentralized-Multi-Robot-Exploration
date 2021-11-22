import numpy as np

from decentralized_exploration.core.constants import Actions


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
