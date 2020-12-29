import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from decentralized_exploration.helpers.plotting import plot_map, plot_grid
from decentralized_exploration.helpers.hex_grid import convert_pixelmap_to_grid


class Robot:
    """
    A class used to represent a single robot

    Instance Attributes
    ----------
    range_finder (decentralized_exploration.core.range_finder.RangeFinder): a RangeFinder object representing the sensor
    width (float) : the width of the robot in meters
    length (float) : the length of the robot in meters
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    hex_map (decentralized_exploration.helpers.hex_grid.Grid): A Grid object holding the hex layer

    Public Methods
    -------
    explore(world): Starts the process of the robot exploring the world. Returns fully explored pixel map
    """

    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.__initialize_map(world_size)

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

        hexagon_size = 4
        self.__pixel_map = -np.ones(world_size)
        self.__hex_map = convert_pixelmap_to_grid(
            self.__pixel_map, hexagon_size)

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
            node_id = self.__hex_map.find_hex(self.__hex_map.hex_at(occ_point)).node_id
            self.__hex_map.update_hex(node_id=node_id, dOccupied=1, dUnknown=-1)

        for free_point in free_points:
            node_id = self.__hex_map.find_hex(self.__hex_map.hex_at(free_point)).node_id
            self.__hex_map.update_hex(node_id=node_id, dFree=1, dUnknown=-1)
        
        self.__hex_map.propagate_rewards()

    def __choose_next_pose(self, current_pos):
        """
        Given the current pos, decides on the next best position for the robot

        Parameters
        ----------
        current_pos (tuple): tuple of integer pixel coordinates

        Returns
        ----------
        desired_hex (Hex): the Hex object representing the desired hex position
        """

        current_hex_id = self.__hex_map.find_hex(self.__hex_map.hex_at(current_pos)).node_id
        interesting_free_hexes = [h.node_id for h in self.__hex_map.all_hexes if h.reward > 0 and h.node_id != current_hex_id]

        if len(interesting_free_hexes) == 0:
            return None

        desired_hex_id = None
        shortest_path = float('inf')

        for h in interesting_free_hexes:
            if nx.has_path(self.__hex_map.graph, source=current_hex_id, target=h):
                path = nx.shortest_path_length(self.__hex_map.graph, source=current_hex_id, target=h)

                if path < shortest_path:
                    shortest_path = path
                    desired_hex_id = h

        desired_hex = self.__hex_map.graph.nodes[desired_hex_id]['hex']

        return desired_hex
    
    def __incremental_pose(self, current_pos, current_orientation, desired_hex):
        """
        Returns the new position and orientation of the robot one increment towards the desired hex.
        If the robot is neighbouring an unknown hex it rotates to see that unknown area. 

        Parameters
        ----------
        current_pos (tuple): tuple of integer pixel coordinates
        current_orientation (int): an int representing the current orientation
        desired_hex (Hex): the Hex object of the desired hex in the current trajectory

        Returns
        ----------
        new_pos (tuple): tuple of integer pixel coordinates of the new position
        new_orientation (int): an int representing the new orientation
        """

        current_hex = self.__hex_map.find_hex(self.__hex_map.hex_at(current_pos))
        current_hex_id = current_hex.node_id

        neighbours = self.__hex_map.hex_neighbours(current_hex)
        unknown_neighbour = False
        for neighbour in neighbours:
            if self.__hex_map.graph.nodes[neighbour]['hex'].state == -1:
                unknown_neighbour = True
                break
        
        if unknown_neighbour:
            next_hex_id = neighbour
        else:
            next_hex_id = nx.shortest_path(self.__hex_map.graph, source=current_hex_id, target=desired_hex.node_id)[1]

        next_hex = self.__hex_map.graph.nodes[next_hex_id]['hex']

        curr_q, curr_r = current_hex.q, current_hex.r
        next_q, next_r = next_hex.q, next_hex.r

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
        else:
            if next_q > curr_q:
                new_orientation = 6
            else:
                new_orientation = 3
        
        if unknown_neighbour:
            new_pos = current_pos
        else:
            new_pos = self.__hex_map.hex_center(next_hex)
            new_pos = np.round(new_pos).astype(int)
        
        return new_pos, new_orientation

    # Public Methods
    def explore(self, world):
        """
        Given the world the robot is exploring, iteratively explores the area

        Parameters
        ----------
        world (decentralized_exploration.core.world.World): a World object that the robot will explore

        Returns
        ----------
        numpy.ndarry: numpy array of pixels representing the fully explored map. 
        """

        # Do a 360 scan
        world.move_robot(world.robot_position, new_orientation=6)
        for i in range(1, 6+1):
            occupied_points, free_points = self.__range_finder.scan(world, new_orientation=i)
            self.__update_map(occupied_points, free_points)

            world.move_robot(world.robot_position, new_orientation=i)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
        plot_grid(self.__hex_map, plot=ax2, robot_pos=world.robot_position)
        plt.pause(1)

        while self.__hex_map.has_unexplored():
            desired_hex = self.__choose_next_pose(world.robot_position)

            if not desired_hex:
                break

            new_position, new_orientation = self.__incremental_pose(world.robot_position, world.robot_orientation, desired_hex)
            print(new_position, new_orientation, world.robot_orientation)

            occupied_points, free_points = self.__range_finder.scan(world, new_orientation)

            self.__update_map(occupied_points, free_points)
            world.move_robot(new_position, new_orientation)

            plot_map(self.__pixel_map, plot=ax1, robot_pos=world.robot_position)
            plot_grid(self.__hex_map, plot=ax2, robot_pos=world.robot_position)
            plt.pause(1)

        return self.__pixel_map
