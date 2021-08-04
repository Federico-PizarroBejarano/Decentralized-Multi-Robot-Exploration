import math
import numpy as np

class Cell:
    """
    A class used to represent a cell in the grid layer

    Instance Attributes
    -------------------
    y (int): the first coordinate
    x (int): the second coordinate
    reward (int): the reward associated with this cell
    V (float): the true value of a cell. Used for displaying computed values.
    probability (float): the probability that another robot will visit a nearby cell
    distance_from_start (int): shortest path length from this cell to the start cell, used to calculate probability
    visited (bool): whether this cell was already visited, used to calculate probability
    previous_cell (Cell): the previous cell in the path to the start cell, used to calculate probability
    state (int): the state of this cell as unknown (-1), free (0), or occupied (1)

    Public Methods
    --------------
    update_cell(nUnknown = 0, nFree = 0, nOccupied = 0): updates the number of unknown, free, 
        and occupied pixels
    """

    # Starting value for nOccupied should not be 0
    def __init__(self, y, x, state=0.0, reward=0.0):
        self.y = y
        self.x = x
        self.state = state
        self.reward = reward
        self.V = 0.0
        self.probability = 0.0
        self.distance_from_start = float('inf')
        self.visited = False
        self.previous_cell = None
    
    # Public Methods
    def update_cell(self, state):
        """
        Updates the cell with changes to the number of unknown, free, and occupied pixels

        Parameters
        ----------
        state (int): the new state of the cell
        """

        self.state = state


class Grid():
    """
    A class used to represent a grid (represented as an dictionary of Cell objects)

    Class Attributes
    ----------------
    radius (int): the radius for which rewards are propagated. A tunable parameter

    Instance Attributes
    -------------------
    all_cells (dictionary): the dictionary of all Cell objects in the grid, indexed by (y, x)

    Static Methods
    --------------
    cell_distance(start_cell, end_cell): returns the distance between two cells 

    Public Methods
    --------------
    add_cell(new_cell): adds a given Cell object to the grid. 
    has_rewards(): returns True if there are Cells with rewards in all_cells
    percent_explored(): returns the percentage of space explored
    cell_neighbours(center_cell): returns list of the adjacent neighbours of given Cell
    propagate_rewards(): updates the reward of every cell
    clear_path(start_cell, end_cell): returns whether there is a clear pixel path between two cell
    find_closest_unknown(center_cell): returns the closest cell that is unknown
    """

    radius = 4

    def __init__(self):
        self.all_cells = {}
    
    # Static Methods
    @staticmethod
    def cell_distance(start_cell, end_cell):
        """
        Takes two cells and returns the distance between them as an integer

        Parameters
        ----------
        start_cell (Cell): a Cell object representing the starting cell
        end_cell (Cell): a Cell object representing the ending cell

        Returns
        -------
        distance (int): a integer representing the Cell distance between two cells
        """

        s_y, s_x = start_cell.y, start_cell.x
        e_y, e_x = end_cell.y, end_cell.x

        distance = max(abs(s_y-e_y), abs(s_x-e_x))
        return int(distance)

    # Public Instance Methods
    def add_cell(self, new_cell):
        """
        Adds a Cell with given axial coordinates into all_cells.

        Parameters
        ----------
        new_cell (Cell): a Cell object with axial coordinates
        """
        
        self.all_cells[(new_cell.y, new_cell.x)] = new_cell
        

    def has_rewards(self):
        """
        Checks whether there are cells left to explore 

        Returns
        -------
        has_rewards (bool): True if there are cells with rewadrs in all_cells, False otherwise
        """

        for cell in self.all_cells.values():
            if cell.reward > 0:
                return True
        
        return False
    
    def percent_explored(self):
        """
        Returns the percentage of cells that are explored

        Returns
        -------
        percent_explored (float): percentage (0-1) of cells explored
        """
        total_cells = len(self.all_cells.values())
        num_unknown = 0.0
        
        for cell in self.all_cells.values():
            if cell.state == -1:
                num_unknown += 1.0
        
        return 1.0 - num_unknown/total_cells

    def cell_neighbours(self, center_cell, radius=1):
        """
        Returns list of neighbours of a given Cell within a specified radius

        Parameters
        ----------
        center_cell (Cell): a Cell object
        radius (int): the radius to consider

        Returns
        -------
        neighbours (list): a list of Cell objects
        """

        neighbours = []

        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                neighbour = self.all_cells[(center_cell.y + y, center_cell.x + x)]
                neighbours.append(neighbour)

        return neighbours

    def clear_path(self, start_cell, end_cell):
        return True

    def propagate_rewards(self):
        """
        Clears the reward from all cells and then re-calculates the reward  at every cell
        """

        for cell in self.all_cells.values():
            cell.reward = 0
        
        for cell in self.all_cells.values():
            if cell.state == -1:
                neighbours = self.cell_neighbours(center_cell=cell, radius=self.radius)

                for neighbour in neighbours:
                    if neighbour.state == 0 and self.clear_path(start_cell=cell, end_cell=cell):
                        neighbour.reward += 1

    
    def find_closest_unknown(self, center_cell):
        """
        If the given cell has a reward, returns one of the unknown cells providing the reward

        Parameters
        ----------
        center_cell (Cell): a Cell object representing the center cell

        Returns
        -------
        unknown_cell (Cell): a Cell representing the neighbouring unknown cell, None if center_cell has no reward
        """

        if center_cell.reward == 0:
            return None
        else:
            neighbours = self.cell_neighbours(center_cell=center_cell, radius=self.radius)

            for neighbour in neighbours:
                if neighbour.state == -1 and self.clear_path(start_cell=center_cell, end_cell=neighbour):
                    return neighbour


def convert_pixelmap_to_grid(pixel_map):
    """
    Converts an image (represented as a numpy.ndarray) into a grid

    Parameters
    ----------
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied

    Returns
    -------
    grid (Grid): a Grid object representing the map
    """

    grid = Grid()

    for y in range(pixel_map.shape[0]):
        for x in range(pixel_map.shape[1]):
            cell = Cell(y, x)
            cell.update_cell(state=pixel_map[y][x])

            grid.add_cell(new_cell=cell)

    return grid


def merge_map(grid, pixel_map, pixel_map_to_merge):
    """
    Merges the current pixel_map with another pixel map.

    Parameters
    ----------
    grid (Grid): a Grid representing the grid layer to be updated
    pixel_map (numpy.ndarry): numpy array of pixels representing the map to be updated
    pixel_map_to_merge (numpy.ndarry): numpy array of pixels representing the map to be merged in 

    Returns
    -------
    pixel_map (numpy.ndarray): the updated pixel_map
    """

    for y in range(pixel_map.shape[0]):
        for x in range(pixel_map.shape[1]):
            if pixel_map[y, x] == -1:
                pixel_map[y, x] = pixel_map_to_merge[y, x]
                found_cell = grid.all_cells[(y, x)]
                found_cell.update_cell(state=pixel_map_to_merge[y, x])

    return pixel_map