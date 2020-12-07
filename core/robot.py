import numpy as np
from helpers import hex_grid

class Robot:
    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.initialize_map(world_size)

    # Getters
    def get_size(self):
        return [self.__width, self.__height]
    
    def get_pixel_map(self):
        return self.__pixel_map
    
    def get_hex_map(self):
        return self.__hex_map

    # Initialize pixel and hex maps
    def initialize_map(self, world_size):
        hexagon_size = 5
        self.__pixel_map = -np.ones(world_size)
        self.__hex_map = hex_grid.convert_image_to_grid(self.__pixel_map, hexagon_size)
    
    def update_map(self, occupied_points, free_points):
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

    def choose_next_pose(self):
        pass

    def explore(self, world):
        while not self.__hex_map.has_unexplored():
            occupied_points, free_points = self.__range_finder.scan(world)
            update_map(occupied_points, free_points)

            new_position, new_orientation = choose_next_pose()
            world.move_robot(new_position, new_orientation)
        
        occupied_points, free_points = self.__range_finder.scan(world)
        self.update_map(occupied_points, free_points)
        return self.__pixel_map
