from decentralized_exploration.core.constants import UNEXPLORED

from decentralized_exploration.core.search.AbstractSearcher import AbstractSearcher
from decentralized_exploration.helpers.grid import Grid

class AStarSearcher(AbstractSearcher):

    def cost(self, neighbour_cell, end_cell):
        return neighbour_cell.distance_from_start + Grid.cell_distance(neighbour_cell, end_cell)

    def terminate(self, current_cell, end_cell):
        return current_cell.state == end_cell 
