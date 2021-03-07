import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state
from decentralized_exploration.helpers.hex_grid import Hex, convert_pixelmap_to_grid
from decentralized_exploration.helpers.plotting import plot_grid


class Robot:
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter
    discount_factor (float): a float less than or equal to 1 that discounts distant values in the MDP
    noise (float): the possibility (between 0 and 1, inclusive), of performing a random action
        rather than the desired action in the MDP
    minimum_change (float): the MDP exits when the largest change in Value is less than this

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
    V = {}

    Public Methods
    --------------
    explore_1_timestep(world): Explores the world for a single timestep/action. 
    """
    # Tunable Parameters
    hexagon_size = 4
    discount_factor = 0.9
    noise = 0.1
    minimum_change = 1

    def __init__(self, robot_id, range_finder, width, length, world_size):
        self.__robot_id = robot_id
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length

        self.__known_robots = {}
        self.__all_states = set()
        self.__V = {}

        self.__initialize_map(world_size=world_size)

    @property
    def robot_id(self):
        return self.__robot_id

    @property
    def size(self):
        return [self.__width, self.__length]

    @property
    def pixel_map(self):
        return self.__pixel_map

    @property
    def hex_map(self):
        return self.__hex_map

    # Private methods
    def __initialize_map(self, world_size):
        """
        Initialized both the internal pixel and hex maps given the size of the world

        Parameters
        ----------
        world_size (tuple): the size of the world map in pixels
        """

        self.__pixel_map = -np.ones(world_size)
        self.__hex_map = convert_pixelmap_to_grid(pixel_map=self.__pixel_map, size=self.hexagon_size)
        
        all_positions = set(self.__hex_map.all_hexes.keys())
        
        for orientation in [1, 2, 3, 4, 5, 6]:
            new_states = set([(position[0], position[1], orientation) for position in all_positions])
            self.__all_states.update(new_states)
        
        self.__V = {state : self.__hex_map.all_hexes[(state[0], state[1])].reward for state in self.__all_states}


    def __update_map(self, occupied_points, free_points):
        """
        Updates both the internal pixel and hex maps given lists of occupied and free pixels

        Parameters
        ----------
        occupied_points (list of [x, y] points): list of occupied points
        free_points (list of [x, y] points): list of free points
        """

        occupied_points = [p for p in occupied_points if self.__pixel_map[p[0], p[1]] == -1]
        free_points = [p for p in free_points if self.__pixel_map[p[0], p[1]] == -1]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.__pixel_map[occ_rows, occ_cols] = 1
        self.__pixel_map[free_rows, free_cols] = 0

        for occ_point in occupied_points:
            desired_hex = self.__hex_map.hex_at(point=occ_point)
            found_hex = self.__hex_map.find_hex(desired_hex=desired_hex)
            self.__hex_map.update_hex(hex_to_update=found_hex, dOccupied=1, dUnknown=-1)

        for free_point in free_points:
            desired_hex = self.__hex_map.hex_at(point=free_point)
            found_hex = self.__hex_map.find_hex(desired_hex=desired_hex)
            self.__hex_map.update_hex(hex_to_update=found_hex, dFree=1, dUnknown=-1)
        
        self.__hex_map.propagate_rewards()


    def __merge_map(self, other_map):
        """
        Merges the current pixel_map with another pixel map.

        Parameters
        ----------
        other_map (numpy.ndarry): numpy array of pixels representing the map to be merged in 
        """

        for y in self.__pixel_map.shape[0]:
            for x in self.__pixel_map.shape[1]:
                if self.__pixel_map[y, x] == -1:
                    self.__pixel_map[y, x] = other_map[y, x]
                    desired_hex = self.__hex_map.hex_at(point=[y, x])
                    found_hex = self.__hex_map.find_hex(desired_hex=desired_hex)

                    if other_map[y, x] == 0:
                        self.__hex_map.update_hex(hex_to_update=found_hex, dFree=1, dUnknown=-1)
                    elif other_map[y, x] == 1:
                        self.__hex_map.update_hex(hex_to_update=found_hex, dOccupied=1, dUnknown=-1)


    def __choose_next_pose(self, current_pos, current_orientation):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_pos (tuple): tuple of integer pixel coordinates
        current_orientation (Orientation): an Orientation object representing current orientation of robot

        Returns
        -------
        new_pos (tuple): tuple of integer pixel coordinates of the new position
        new_orientation (Orientation): an Orientation object representing the new orientation
        is_clockwise (bool): a bool representing whether the rotation is clockwise
        """

        current_hex_pos = self.__hex_map.hex_at(point=current_pos)
        current_hex = self.__hex_map.find_hex(desired_hex=current_hex_pos)
        current_state = (current_hex.q, current_hex.r, current_orientation)
        
        # Checking if on reward hexagon
        on_reward_hex = current_hex.reward > 0
        
        if on_reward_hex:
            next_hex = self.__hex_map.find_closest_unknown(center_hex=current_hex)
            is_clockwise = find_new_orientation(current_hex=current_hex, current_orientation=current_orientation, next_hex=next_hex)
            action = Actions.CLOCKWISE if is_clockwise else Actions.COUNTER_CLOCKWISE
            next_state = get_new_state(current_state, action)
            return next_state

        # Initial policy
        rewards = { key:hexagon.reward for (key, hexagon) in self.__hex_map.all_hexes.items() }

        policy = {}

        biggest_change = float('inf')

        while biggest_change >= self.minimum_change:
            biggest_change = 0
            
            for state in self.__all_states:
                if self.__hex_map.all_hexes[(state[0], state[1])].state != 0:
                    continue

                old_value = self.__V[state]
                new_value = 0
                
                for action in possible_actions(state, self.__hex_map):
                    next_state = get_new_state(state, action)

                    # Choose a random action to do with probability self.noise
                    random_action = np.random.choice([rand_act for rand_act in possible_actions(state, self.__hex_map) if rand_act != action])
                    random_state = get_new_state(state, random_action)

                    value = rewards[(state[0], state[1])] + self.discount_factor * ((1 - self.noise)* self.__V[next_state] + (self.noise * self.__V[random_state]))
                    
                    # Keep best action so far
                    if value > new_value:
                        new_value = value
                        policy[state] = action

                # Save best value                         
                self.__V[state] = new_value
                biggest_change = max(biggest_change, np.abs(old_value - self.__V[state]))

        next_state = get_new_state(state=current_state, action=policy[current_state])

        return next_state

    # Public Methods
    def complete_rotation(self, world):
        """
        Rotates the robot completely to scan the area around it

        Parameters
        ----------
        world (World): a World object that the robot will explore
        """

        starting_orientation = world.get_orientation(self.__robot_id)
        next_orientation = starting_orientation.rotate_clockwise()

        while starting_orientation != next_orientation:
            occupied_points, free_points = self.__range_finder.scan(world=world, position=world.get_position(self.__robot_id), old_orientation=world.get_orientation(self.__robot_id), new_orientation=next_orientation, is_clockwise=False)
            self.__update_map(occupied_points=occupied_points, free_points=free_points)

            world.move_robot(robot_id=self.__robot_id, new_position=world.get_position(self.__robot_id), new_orientation=next_orientation)
            next_orientation.rotate_clockwise(in_place=True)


    def communicate(self, message):
        """
        Communicates with the other robots in the team. Receives a message and updates the 
        last known position and last updated time of every robot that transmitted a message. 
        Additionally, merges in all their pixel maps.

        Parameters
        ----------
        message (dict): a dictionary containing the robot position and pixel map of the other robots
        """

        for robot_id in message:
            self.__known_robots[robot_id] = {
                'last_updated': datetime.now(),
                'last_known_position': message[robot_id].robot_position
            }

            self.__merge_map(other_map=message[robot_id].pixel_map)


    def explore_1_timestep(self, world):
        """
        Given the world the robot is exploring, explores the area for 1 timestep/action

        Parameters
        ----------
        world (World): a World object that the robot will explore
        """

        new_state = self.__choose_next_pose(current_pos=world.get_position(self.__robot_id), current_orientation=world.get_orientation(self.__robot_id))
        new_position = self.__hex_map.hex_center(Hex(new_state[0], new_state[1]))
        new_position = [int(coord) for coord in new_position]
        new_orientation = new_state[2]

        occupied_points, free_points = self.__range_finder.scan(world=world, position=world.get_position(self.__robot_id), old_orientation=world.get_orientation(self.__robot_id), new_orientation=new_orientation)

        self.__update_map(occupied_points=occupied_points, free_points=free_points)
        world.move_robot(robot_id=self.__robot_id, new_position=new_position, new_orientation=new_orientation)
