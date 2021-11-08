from heapdict import heapdict
from abc import ABCMeta, abstractmethod

from decentralized_exploration.core.constants import UNOCCUPIED

class AbstractSearcher:
    
    __metaclass__ = ABCMeta

    def initialize_grid_for_search(self, start_cell, grid):
        for cell in grid.all_cells.values():
            cell.distance_from_start = float('inf')
            cell.visited = False
            cell.previous_cell = None
            
        start_cell.visited = True
        start_cell.distance_from_start = 0

    @abstractmethod
    def cost(self, neighbour_cell, end_cell):
        pass

    @abstractmethod
    def terminate(current_cell, end_cell):
        pass


    def initialize_cell_queue(self, start_cell, end_cell, grid, robot_states):
        cell_queue = heapdict() 

        other_robots = [robot_states[robot].pixel_position for robot in robot_states if robot_states[robot].pixel_position != start_cell.coord]

        neighbour_cells = grid.cell_neighbours(center_cell=start_cell, radius=1)
        for neighbour_cell in neighbour_cells:
            if neighbour_cell.state == 0: 
                neighbour_cell.distance_from_start = start_cell.distance_from_start + 1 
                neighbour_cell.previous_cell = start_cell
                
                cell_queue[neighbour_cell] = self.cost(neighbour_cell, end_cell)

                # Do not explore the neighbours with a robot in them
                if neighbour_cell.coord in other_robots:
                    cell_queue.popitem()

        return cell_queue 


    def search(self, start_cell, end_cell, grid, robot_states):
        '''
        Uses Dijkstra's algorithm to find the nearest free, frontier Cell.

        Parameters
        ----------
        start_cell (Cell): the starting Cell position
        grid (Grid): the Grid object representing the grid 

        Returns
        -------
        cell (tuple): the closest frontier Cell
        ''' 

        self.initialize_grid_for_search(start_cell, grid)
        cell_queue = self.initialize_cell_queue(start_cell, end_cell, grid, robot_states)
        
        while(len(cell_queue) != 0):
            current_cell = cell_queue.popitem()[0]
            if current_cell.state == UNOCCUPIED:
                for neighbour_cell in grid.cell_neighbours(current_cell):
                    new_distance_from_start = current_cell.distance_from_start + 1
                    if new_distance_from_start < neighbour_cell.distance_from_start:
                        neighbour_cell.previous_cell = current_cell
                        neighbour_cell.distance_from_start = new_distance_from_start 
                        cell_queue[neighbour_cell] = self.cost(neighbour_cell, end_cell) 

                    # Terminate when we found a frontier cell (a cell that neighbours an unxplored cell)
                    if self.terminate(neighbour_cell, end_cell): 
                        return current_cell
        return None


    def get_next_cell(self, start_cell, end_cell):
        next_cell = end_cell
        
        while next_cell.previous_cell != start_cell and next_cell.previous_cell != None:
            next_cell = next_cell.previous_cell

        return next_cell