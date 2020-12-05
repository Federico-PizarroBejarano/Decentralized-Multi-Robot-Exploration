import numpy as np

class Robot:
    def __init__(self, range_finder, width, length, world_size):
        self.__range_finder = range_finder
        self.__width = width
        self.__length = length
        self.initialize_map(world_size)

    # Getters
    def get_size(self):
        return [self.__width, self.__height]

    def initialize_map(self, world_size):
        self.__internal_map = np.ones(world_size)
    
    def update_map(self, distance):
        pass

    def choose_next_pose(self):
        pass

    def explore(self, world):
        while np.any(self.__internal_map == 2):
            distance = self.__range_finder.scan(world)
            update_map(distance)

            new_position, new_orientation = choose_next_pose()
            world.move_robot(new_position, new_orientation)
        
        return self.__internal_map
