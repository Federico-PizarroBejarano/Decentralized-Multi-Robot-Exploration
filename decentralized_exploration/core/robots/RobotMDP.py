import numpy as np

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import get_new_state, solve_MDP, compute_probability, closest_reward, path_between_cells, max_value, possible_actions
from decentralized_exploration.helpers.grid import Cell, Grid, merge_map


class RobotMDP(AbstractRobot):
    '''
    A class used to represent a single robot

    Class Attributes
    ----------------
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this
    minimum_change_repulsive (float): the repulsive MDP exits when the largest change in Value is less than this
    max_iterations (int): the maximum number of iterations before the MDP returns
    rho (float): reward set at positions of other robots when calculating repulsive MDP
    horizon (int): how near a state is from the current state to be considered in the MDP
    exploration_horizon (int): how near a state is from another robot to be considered a possible future location
        when calculating probability
    weighing_factor (float): weighs the effect of the DVF on the final MDP
    '''

    # Tunable Parameters
    discount_factor = 0.8
    noise = 0.0
    minimum_change = 1.0
    minimum_change_repulsive = 0.01
    max_iterations = 50
    rho = 0.10
    horizon = 20
    exploration_horizon = 4
    weighing_factor = 4.0

    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotMDP, self).__init__(robot_id, range_finder, width, length, world_size)
        self._all_states = set()
        self._V = {}
        self._repulsive_V = {}
        self._initialize_values()


    # Private methods
    def _initialize_values(self):
        '''
        Initializes the set of states and the initial value function of the robot
        '''

        self._all_states = set(self.grid.all_cells.keys())
        
        self._V = {state : self.grid.all_cells[state].reward for state in self._all_states}
        self._repulsive_V = {state : 0 for state in self._all_states}
    

    def _calculate_V(self, current_robot, horizon, robot_states):
        '''
        Calculates the probable value function for another robot

        Parameters
        ----------
        current_robot (str): the robot_id of the robot whose value function will be estimated
        horizon (int): how near a state is from the current state to be considered in the MDP
        '''

        current_cell = self.grid.all_cells[self._known_robots[current_robot]['last_known_position']]

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.cell_distance(self.grid.all_cells[robot['last_known_position']], current_cell) < horizon and robot_id != current_robot]            
        close_robot_states = [self._known_robots[robot]['last_known_position'] for robot in close_robots]

        initial_repulsive_rewards = { key: self.rho if key in close_robot_states else 0 for key in self.grid.all_cells.keys() }

        self._known_robots[current_robot]['repulsive_V'] = { key: self.rho if key in close_robot_states else 0 for key in self.grid.all_cells.keys() }
        solve_MDP(self.grid, self._known_robots[current_robot]['repulsive_V'], initial_repulsive_rewards, self.noise, self.discount_factor, self.minimum_change_repulsive, self.max_iterations, horizon, horizon, current_cell, robot_states)

        repulsive_reward = { key:0 for key in self.grid.all_cells.keys() }
    
        for state in self._all_states:
            repulsive_reward[state] += self._known_robots[current_robot]['repulsive_V'][state]

        rewards = { key:cell.reward - repulsive_reward[key] for (key, cell) in self.grid.all_cells.items() }

        self._known_robots[current_robot]['V'] = {state : self.grid.all_cells[state].reward for state in self._all_states}

        closest_reward_cell = closest_reward(current_cell=current_cell, grid=self.grid, robot_states=robot_states)[1]
        if closest_reward_cell == None:
            min_iterations = self.max_iterations // 2
        else:
            min_iterations = closest_reward_cell.distance_from_start

        modified_discount_factor = 1.0 - 80.0/(max(self.horizon, min_iterations)**2.0)
        solve_MDP(self.grid, self._known_robots[current_robot]['V'], rewards, self.noise, modified_discount_factor, self.minimum_change, self.max_iterations, min_iterations, horizon, current_cell, robot_states)


    def _compute_DVF(self, current_cell, iteration, horizon, robot_states):
        '''
        Updates the repulsive value at each state. This is then used in _choose_next_pose to avoid other robots

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        horizon (int): how near a state is from the current state to be considered in the MDP
        '''

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.cell_distance(self.grid.all_cells[robot['last_known_position']], current_cell) < horizon and robot_id != self.robot_id]

        DVF = {state : 0 for state in self._all_states}

        for robot in close_robots:
            if self._known_robots[robot]['last_updated'] == iteration:
                self._calculate_V(current_robot=robot, horizon=horizon, robot_states=robot_states)
            
            robot_cell = self.grid.all_cells[self._known_robots[robot]['last_known_position']]
            compute_probability(start_cell=robot_cell,
                                time_increment=iteration - self._known_robots[robot]['last_updated'],
                                exploration_horizon=self.exploration_horizon,
                                grid=self.grid)

            for state in self._all_states:
                DVF[state] += self.weighing_factor * self.grid.all_cells[state].probability * abs(self._known_robots[robot]['V'][state]) + self.weighing_factor**3 * self.grid.all_cells[state].probability**2
        
        for state in self._all_states:
            DVF[state] = abs(DVF[state])

        return DVF


    def _choose_next_pose(self, current_position, iteration, robot_states):
        '''
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm

        Returns
        -------
        next_state (tuple): tuple of q and r coordinates of the new position
        '''

        current_cell = self.grid.all_cells[current_position]

        closest_reward_cell, max_path_distance = closest_reward(current_cell=current_cell, grid=self.grid, robot_states=robot_states)[1:]
        
        if closest_reward_cell == None:
            min_iterations = self.max_iterations // 2
            modified_horizon = self.horizon
        else:
            min_iterations = closest_reward_cell.distance_from_start
            modified_horizon = max(self.horizon, max_path_distance+1)

        DVF = self._compute_DVF(current_cell=current_cell, iteration=iteration, horizon=modified_horizon, robot_states=robot_states)

        rewards = { key:cell.reward for (key, cell) in self.grid.all_cells.items() if Grid.cell_distance(cell, current_cell) < modified_horizon + 1}

        max_DVF = max(DVF.values())
        max_reward = max([ reward for reward in rewards.values() ]) + 0.01
        cumulative_DVF = sum(DVF.values())
        cumulative_reward = sum([ reward for reward in rewards.values() ]) + 0.01

        reward_multiplier = max(2, 2*max_DVF/max_reward, 2*cumulative_DVF/cumulative_reward)        
        modified_discount_factor = 1.0 - 80.0/(max(self.horizon, min_iterations)**2.0)

        # print('Reward multiplier: {}, Modified gamma: {}'.format(reward_multiplier, modified_discount_factor))

        self._V = {state : self.grid.all_cells[state].reward * reward_multiplier for state in self._all_states}

        solve_MDP(self.grid, self._V, rewards, self.noise, modified_discount_factor, self.minimum_change, self.max_iterations, min_iterations, modified_horizon, current_cell, robot_states, DVF)

        action = max_value(self._V, self._grid, current_position, possible_actions(state=current_position, grid=self._grid, robot_states=robot_states))
        next_state = get_new_state(state=current_position, action=action)

        if abs(self._V[current_position] - self._V[next_state]) < 0.001:
            print(self.robot_id, ' MDP is stuck, using greedy')
            next_position = closest_reward(current_cell=current_cell, grid=self.grid, robot_states=robot_states)[0]

            # All rewards have been found
            if next_position == None:
                return current_position

            next_cell = Cell(next_position[0], next_position[1])

            next_state = path_between_cells(current_cell=current_cell, goal_cell=next_cell, grid=self.grid, robot_states=robot_states)
        
        # Plotting
        for state in self._all_states:
            self.grid.all_cells[state].V = 0

        for state in self._all_states:
            if Grid.cell_distance(current_cell, Cell(state[0], state[1])) < modified_horizon + 1:
                self.grid.all_cells[state].V += self._V[state]

        return next_state


    # Public Methods
    def communicate(self, message, iteration):
        '''
        Communicates with the other robots in the team. Receives a message and updates the 
        last known position and last updated time of every robot that transmitted a message. 
        Additionally, merges in all their pixel maps.

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        iteration (int): the current iteration
        '''

        for robot_id in message:
            if robot_id not in self._known_robots:
                self._known_robots[robot_id] = {
                    'V': {state : self.grid.all_cells[state].reward for state in self._all_states},
                    'repulsive_V': {state : 0 for state in self._all_states}
                }
            
            self._known_robots[robot_id]['last_updated'] = iteration
            self._known_robots[robot_id]['last_known_position'] = message[robot_id]['robot_position']

            self.__pixel_map = merge_map(grid=self.grid, pixel_map=self.pixel_map, pixel_map_to_merge=message[robot_id]['pixel_map'])
            self.grid.propagate_rewards()

        self._known_robots[self.robot_id] = {
            'last_updated': iteration,
            'V': self._V,
            'repulsive_V': self._repulsive_V
        }