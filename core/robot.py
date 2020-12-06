import numpy as np
from helpers import hex

class Robot:
    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.initialize_map(world_size)

    # Getters
    def get_size(self):
        return [self.__width, self.__height]

    # Initialize pixel and hex maps
    def initialize_map(self, world_size):
        self.__internal_pixel_map = np.ones(world_size)
        self.__internal_hex_map = hex.convert_image_to_grid(self.__internal_pixel_map)
    
    def update_map(self, distance):
        pass

    def choose_next_pose(self):
        pass

    def explore(self, world):
        while self.__internal_hex_map.has_unexplored():
            distance = self.__range_finder.scan(world)
            update_map(distance)

            new_position, new_orientation = choose_next_pose()
            world.move_robot(new_position, new_orientation)
        
        return self.__internal_pixel_map
