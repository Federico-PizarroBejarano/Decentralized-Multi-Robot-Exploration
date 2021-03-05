import numpy as np
import matplotlib.pyplot as plt

from decentralized_exploration.core.constants import Actions
from decentralized_exploration.helpers.decision_making import find_new_orientation, possible_actions, get_new_state
from decentralized_exploration.helpers.hex_grid import Hex, convert_pixelmap_to_grid
from decentralized_exploration.helpers.plotting import plot_map, plot_grid


class Robot:
    """
    A class used to represent a single robot

    Class Attributes
    ----------------
    hexagon_size (int): the size of the hexagons compared to each pixel. A tunable parameter

    Instance Attributes
    -------------------
    range_finder (RangeFinder): a RangeFinder object representing the sensor
    width (float) : the width of the robot in meters
    length (float) : the length of the robot in meters
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    hex_map (Grid): A Grid object holding the hex layer

    Public Methods
    --------------
    explore(world): Starts the process of the robot exploring the world. Returns fully explored pixel map
    """
    # Tunable Parameters
    hexagon_size = 4
    discount_factor = 0.9
    noise = 0.1
    minimum_change = 1

    all_states = set()
    V = {}

    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.__initialize_map(world_size=world_size)

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
            self.all_states.update(new_states)
        
        self.V = {state : self.__hex_map.all_hexes[(state[0], state[1])].reward for state in self.all_states}

    def __update_map(self, occupied_points, free_points):
        """
        Updates both the internal pixel and hex maps given arrays of occupied and free pixels

        Parameters
        ----------
        occupied_points (array of [x, y] points): array of occupied points
        free_points (array of [x, y] points): array of free points
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

    def __choose_next_pose(self, current_pos, current_orientation):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_pos (tuple): tuple of integer pixel coordinates
        current_orientation (int): int representing current orientation of robot

        Returns
        -------
        new_pos (tuple): tuple of integer pixel coordinates of the new position
        new_orientation (int): an int representing the new orientation
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
            
            for state in self.all_states:
                if self.__hex_map.all_hexes[(state[0], state[1])].state != 0:
                    continue

                old_value = self.V[state]
                new_value = 0
                
                for action in possible_actions(state, self.__hex_map):
                    next_state = get_new_state(state, action)

                    # Choose a random action to do with probability self.noise
                    random_action = np.random.choice([rand_act for rand_act in possible_actions(state, self.__hex_map) if rand_act != action])
                    random_state = get_new_state(state, random_action)

                    value = rewards[(state[0], state[1])] + self.discount_factor * ((1 - self.noise)* self.V[next_state] + (self.noise * self.V[random_state]))
                    
                    # Keep best action so far
                    if value > new_value:
                        new_value = value
                        policy[state] = action

                # Save best value                         
                self.V[state] = new_value
                biggest_change = max(biggest_change, np.abs(old_value - self.V[state]))

        next_state = get_new_state(state=current_state, action=policy[current_state])

        return next_state

    # Public Methods
    def explore(self, world):
        """
        Given the world the robot is exploring, iteratively explores the area

        Parameters
        ----------
        world (World): a World object that the robot will explore

        Returns
        -------
        pixel_map (numpy.ndarry): numpy array of pixels representing the fully explored map. 
        """

        fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(111)

        # Do a 360 scan
        world.move_robot(new_position=world.robot_position, new_orientation=6)
        for orientation in range(1, 6+1):
            occupied_points, free_points = self.__range_finder.scan(world=world, new_orientation=orientation, is_clockwise=False)
            self.__update_map(occupied_points=occupied_points, free_points=free_points)

            world.move_robot(new_position=world.robot_position, new_orientation=orientation)
            # plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
            plot_grid(grid=self.__hex_map, plot=ax2, robot_pos=world.robot_position, robot_orientation=world.robot_orientation)
            plt.pause(0.05)

        while self.__hex_map.has_unexplored():
            new_state = self.__choose_next_pose(current_pos=world.robot_position, current_orientation=world.robot_orientation)
            new_position = self.__hex_map.hex_center(Hex(new_state[0], new_state[1]))
            new_position = [int(coord) for coord in new_position]
            new_orientation = new_state[2]

            occupied_points, free_points = self.__range_finder.scan(world=world, new_orientation=new_orientation)

            self.__update_map(occupied_points=occupied_points, free_points=free_points)
            world.move_robot(new_position=new_position, new_orientation=new_orientation)

            # plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
            plot_grid(grid=self.__hex_map, plot=ax2, robot_pos=world.robot_position, robot_orientation=world.robot_orientation)
            plt.pause(0.05)

        return self.__pixel_map
