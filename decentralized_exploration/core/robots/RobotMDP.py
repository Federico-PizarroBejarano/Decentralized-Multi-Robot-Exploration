import numpy as np

from decentralized_exploration.core.robots.AbstractRobot import AbstractRobot
from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state, solve_MDP, compute_probability
from decentralized_exploration.helpers.hex_grid import Hex, Grid, merge_map


class RobotMDP(AbstractRobot):
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this
    minimum_change_repulsive (float): the repulsive MDP exits when the largest change in Value is less than this
    max_iterations (int): the maximum number of iterations before the MDP returns
    horizon (int): how near a state is from the current state to be considered in the MDP

    Instance Attributes
    -------------------
    robot_id (str): the unique id of this robot
    range_finder (RangeFinder): a RangeFinder object representing the sensor
    width (float) : the width of the robot in meters
    length (float) : the length of the robot in meters
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    hex_map (Grid): A Grid object holding the hex layer
    known_robots (dict): a dictionary containing the last known position and last time of contact
        with every other known robot in the team
    all_states (list)
    V (dict): a dictionary containing the value at each state, indexed by state
    repulsive_V (dict): a dictionary containing the repulsive value at each state, indexed by state

    Public Methods
    --------------
    explore_1_timestep(world): Explores the world for a single timestep/action. 
    """

    # Tunable Parameters
    discount_factor = 0.9
    noise = 0.1
    minimum_change = 0.1
    minimum_change_repulsive = 0.1
    max_iterations = 50
    rho = 0.10
    horizon = 15
    exploration_horizon = 5
    weighing_factor = 15.0

    def __init__(self, robot_id, range_finder, width, length, world_size):
        super(RobotMDP, self).__init__(robot_id, range_finder, width, length, world_size)
        self._all_states = set()
        self._V = {}
        self._repulsive_V = {}
        self._initialize_values()


    # Private methods
    def _initialize_values(self):
        """
        Initializes the set of states and the initial value function of the robot
        """
                
        for orientation in [1, 2, 3, 4, 5, 6]:
            new_states = set([(position[0], position[1], orientation) for position in self.hex_map.all_hexes.keys()])
            self._all_states.update(new_states)
        
        self._V = {state : self.hex_map.all_hexes[(state[0], state[1])].reward for state in self._all_states}
    

    def _calculate_V(self, current_robot):
            current_hex_pos = self.hex_map.hex_at(point=self._known_robots[current_robot]['last_known_position'])
            current_hex = self.hex_map.find_hex(desired_hex=current_hex_pos)

            close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.hex_distance(self.hex_map.hex_at(robot['last_known_position']), current_hex_pos) < self.horizon and robot_id != current_robot]            
            close_robot_states = [self.hex_map.hex_at(self._known_robots[robot]['last_known_position']) for robot in close_robots]
            close_robot_states = [(hex_position.q, hex_position.r) for hex_position in close_robot_states]

            initial_repulsive_rewards = { key: self.rho if key in close_robot_states else 0 for key in self.hex_map.all_hexes.keys() }

            self._known_robots[current_robot]['repulsive_V'] = {state : 0 for state in self._all_states}
            solve_MDP(self.hex_map, self._known_robots[current_robot]['repulsive_V'], initial_repulsive_rewards, self.noise, self.discount_factor, self.minimum_change_repulsive, self.max_iterations, self.horizon, current_hex)

            repulsive_reward = { key:0 for key in self.hex_map.all_hexes.keys() }
        
            for state in self._all_states:
                repulsive_reward[(state[0], state[1])] += self._known_robots[current_robot]['repulsive_V'][state]

            rewards = { key:hexagon.reward - repulsive_reward[key] for (key, hexagon) in self.hex_map.all_hexes.items() }

            self._known_robots[current_robot]['V'] = {state : self.hex_map.all_hexes[(state[0], state[1])].reward for state in self._all_states}
            solve_MDP(self.hex_map, self._known_robots[current_robot]['V'], rewards, self.noise, self.discount_factor, self.minimum_change, self.max_iterations, self.horizon, current_hex)


    def _compute_DVF(self, current_hex, iteration):
        """
        Updates the repulsive value at each state. This is then used in _choose_next_pose to avoid other robots

        Parameters
        ----------
        current_position (tuple): tuple of integer pixel coordinates
        iteration (int): the current iteration of the algorithm
        """

        close_robots = [robot_id for (robot_id, robot) in self._known_robots.items() if Grid.hex_distance(self.hex_map.hex_at(robot['last_known_position']), current_hex) < self.horizon and robot_id != self.robot_id]

        DVF = {state : 0 for state in self._all_states}

        for robot in close_robots:
            if self._known_robots[robot]['last_updated'] == iteration:
                self._calculate_V(current_robot=robot)
            
            robot_hex = self.hex_map.find_hex(self.hex_map.hex_at(point=self._known_robots[robot]['last_known_position']))
            compute_probability(start_hex=robot_hex,
                                time_increment=iteration - self._known_robots[robot]['last_updated'],
                                exploration_horizon=self.exploration_horizon,
                                hex_map=self.hex_map)

            for state in self._all_states:
                DVF[state] +=  self.weighing_factor * self.hex_map.all_hexes[(state[0], state[1])].probability * self._known_robots[robot]['V'][state]
        
        for state in self._all_states:
            DVF[state] = abs(DVF[state])

        return DVF


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
            return next_state

        DVF = self._compute_DVF(current_hex=current_hex_pos, iteration=iteration)
        
        rewards = { key:hexagon.reward for (key, hexagon) in self.hex_map.all_hexes.items() }
        
        self._V = {state : self.hex_map.all_hexes[(state[0], state[1])].reward for state in self._all_states}
        policy = solve_MDP(self.hex_map, self._V, rewards, self.noise, self.discount_factor, self.minimum_change, self.max_iterations, self.horizon, current_hex, DVF)
        
        next_state = get_new_state(state=current_state, action=policy[current_state])

        # Plotting
        for state in self._all_states:
            self.hex_map.find_hex(Hex(state[0], state[1])).V = 0

        for state in self._all_states:
            self.hex_map.find_hex(Hex(state[0], state[1])).V += self._V[state]

        return next_state

