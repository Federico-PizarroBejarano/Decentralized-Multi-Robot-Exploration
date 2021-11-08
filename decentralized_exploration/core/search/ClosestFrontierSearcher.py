from decentralized_exploration.core.constants import UNEXPLORED

from decentralized_exploration.core.search.AbstractSearcher import AbstractSearcher

class ClosestFrontierSearcher(AbstractSearcher):

    def cost(self, neighbour_cell, end_cell):
        return neighbour_cell.distance_from_start 

    def terminate(self, current_cell, end_cell):
        return current_cell.state == UNEXPLORED
