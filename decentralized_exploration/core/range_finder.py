import numpy as np

class RangeFinder:
    def __init__(self, distance, frequency):
        self.__distance = distance
        self.__frequency = frequency

    # Getters
    def get_distance(self):
        return self.__distance
    
    def get_frequency(self):
        return self.__frequency
    
    # Scans forward
    def scan(self, world):
        return world.scan(self)