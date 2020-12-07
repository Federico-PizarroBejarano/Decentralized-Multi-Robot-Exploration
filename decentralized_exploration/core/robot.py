import numpy as np
import networkx as nx

from decentralized_exploration.helpers.plotting import plot_map, plot_grid
from decentralized_exploration.helpers.hex_grid import convert_image_to_grid

class Robot:
    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.initialize_map(world_size)

    # Getters
    def get_size(self):
        return [self.__width, self.__length]
    
    def get_pixel_map(self):
        return self.__pixel_map
    
    def get_hex_map(self):
        return self.__hex_map

    # Initialize pixel and hex maps
    def initialize_map(self, world_size):
        hexagon_size = 5
        self.__pixel_map = -np.ones(world_size)
        self.__hex_map = convert_image_to_grid(self.__pixel_map, hexagon_size)
    
    def update_map(self, occupied_points, free_points):
        occupied_points = [p for p in occupied_points if self.__pixel_map[p[0], p[1]] == -1]
        free_points = [p for p in free_points if self.__pixel_map[p[0], p[1]] == -1]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points] 
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.__pixel_map[occ_rows, occ_cols] = 1
        self.__pixel_map[free_rows, free_cols] = 0

        for occ_point in occupied_points:
            node_id = self.__hex_map.find_hex(self.__hex_map.hex_at(occ_point)).node_id
            self.__hex_map.update_hex(node_id, nOccupied = 1, nUnknown = -1)
        
        for free_point in free_points:
            node_id = self.__hex_map.find_hex(self.__hex_map.hex_at(free_point)).node_id
            self.__hex_map.update_hex(node_id, nFree = 1, nUnknown = -1)

    def choose_next_pose(self, current_pose):
        unexplored_hexes = [h for h in self.__hex_map.allHexes if h.state == -1]
        interesting_free_hexes = set()

        for h in unexplored_hexes:
            neighbours =  self.__hex_map.hex_neighbours(h)
            if neighbours:
                free_neighbours = [n[0] for n in neighbours if n[1] == 0]
                if len(free_neighbours) > 0:
                    interesting_free_hexes = interesting_free_hexes.union(set(free_neighbours))

        if len(interesting_free_hexes) == 0:
            return []

        closest_hex_id = None
        current_hex_id = self.__hex_map.find_hex(self.__hex_map.hex_at(current_pose)).node_id
        shortest_path = float('inf')

        for h in interesting_free_hexes:
            if nx.has_path(self.__hex_map.graph, current_hex_id, h):
                path = nx.shortest_path_length(self.__hex_map.graph, current_hex_id, h)

                if path < shortest_path: 
                    shortest_path = path
                    closest_hex_id = h

        closest_hex = self.__hex_map.graph.nodes[closest_hex_id]['hex']
        robot_pose = self.__hex_map.hex_center(closest_hex)

        return np.round(robot_pose).astype(int)


    def explore(self, world):
        while self.__hex_map.has_unexplored():
            occupied_points, free_points = self.__range_finder.scan(world)
            self.update_map(occupied_points, free_points)

            plot_map(self.__pixel_map, world.get_position())
            plot_grid(self.__hex_map, world.get_position())

            new_position = self.choose_next_pose(world.get_position())
            if len(new_position) == 0:
                break

            world.move_robot(new_position, new_orientation = 1)
        
        occupied_points, free_points = self.__range_finder.scan(world)
        self.update_map(occupied_points, free_points)
        return self.__pixel_map
