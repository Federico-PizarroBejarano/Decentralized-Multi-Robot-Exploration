import numpy as np
from helpers.field_of_view import field_of_view

class World:
    def __init__(self, world_map, pixel_size, robot, robot_position, robot_orientation):
        self.__map = world_map
        self.__pixel_size = pixel_size
        self.__robot = robot
        self.__robot_pos = robot_position
        self.__robot_orientation = robot_orientation
    
    # Getters
    def get_position(self):
        return self.__robot_pos
    
    def get_orientation(self):
        return self.__robot_orientation
    
    # Robot moves
    def move_robot(self, new_position, new_orientation):
        # Check move is possible
        # Check there is no collision

        self.__robot_pos = new_position
        self.__robot_orientation = new_orientation

        return True

    # Scans forward
    def scan(self, range_finder):
        return field_of_view(self.__map, self.__robot_pos, self.__map.shape)
    
