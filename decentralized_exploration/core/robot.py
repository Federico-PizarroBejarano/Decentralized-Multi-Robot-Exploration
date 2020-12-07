import numpy as np
import networkx as nx

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
        self.__hex_map = convert_pixelmap_to_grid(self.__pixel_map, hexagon_size)
    

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
            self.__hex_map.update_hex(node_id = node_id, dOccupied = 1, dUnknown = -1)
        
        for free_point in free_points:
            node_id = self.__hex_map.find_hex(self.__hex_map.hex_at(free_point)).node_id
            self.__hex_map.update_hex(node_id = node_id, dFree = 1, dUnknown = -1)


    def __choose_next_pose(self, current_pose):
        """
        Given the current pose, decides on the next best position for the robot

        Parameters
        ----------
        current_pose (tuple): tuple of integer pixel coordinates
        
        Returns
        ----------
        list: 2-element list of pixel coordinates
        """

        unexplored_hexes = [h for h in self.__hex_map.allHexes if h.state == -1]
        interesting_free_hexes = set()

        for h in unexplored_hexes:
            neighbours =  self.__hex_map.hex_neighbours(h)
            if neighbours:
                free_neighbours = [n for n in neighbours if self.__hex_map.graph.nodes[n]['hex'].state == 0]
                if len(free_neighbours) > 0:
                    interesting_free_hexes = interesting_free_hexes.union(set(free_neighbours))

        if len(interesting_free_hexes) == 0:
            return []

        closest_hex_id = None
        current_hex_id = self.__hex_map.find_hex(self.__hex_map.hex_at(current_pose)).node_id
        shortest_path = float('inf')

        for h in interesting_free_hexes:
            if nx.has_path(self.__hex_map.graph, source=current_hex_id, target=h):
                path = nx.shortest_path_length(self.__hex_map.graph, source=current_hex_id, target=h)

                if path < shortest_path: 
                    shortest_path = path
                    closest_hex_id = h

        closest_hex = self.__hex_map.graph.nodes[closest_hex_id]['hex']
        robot_pose = self.__hex_map.hex_center(closest_hex)

        return np.round(robot_pose).astype(int)


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

        while self.__hex_map.has_unexplored():
            occupied_points, free_points = self.__range_finder.scan(world)
            self.__update_map(occupied_points, free_points)

            plot_map(self.__pixel_map, world.robot_position)
            plot_grid(self.__hex_map, world.robot_position)

            new_position = self.__choose_next_pose(world.robot_position)
            if len(new_position) == 0:
                break

            world.move_robot(new_position, new_orientation = 1)
        
        return self.__pixel_map
