import math
import numpy as np
import networkx as nx

class Hex:
    """
    A class used to represent a hexagon in the hexagonal grid layer

    Class Attributes
    ----------
    Tr(int): the ratio between free and unknown pixels required for the hexagon
        to be considered free. A tunable parameter

    Instance Attributes
    ----------
    q(int): the first axial coordinate
    r(int): the second axial coordinate
    s(int): the third axial coordinate satisfying (q + r + s = 0)
    node_id(int): the id of this hexagon in the graph
    nUnknown(int): number of unknown pixels in this hexagon
    nFreeint): number of free pixels in this hexagon
    nOccupied(int): number of occupied pixels in this hexagon
    state(int): the state of this hexagon as unknown (-1), free (0), or occupied (1)
    """

    # Tunable Parameter
    Tr = 0.5

    # Starting value for nOccupied should not be 0
    def __init__(self, q, r, node_id = -1, nUnknown = 0, nFree = 0, nOccupied = 0):
        self.q = q
        self.r = r
        self.node_id = node_id
        self.nUnknown = nUnknown
        self.nFree = nFree
        self.nOccupied = nOccupied

    @property
    def s(self):
        return -(self.q + self.r)
    
    @property
    def state(self):
        if self.nOccupied > 0:
            return 1
        elif self.nUnknown == 0:
            return 0
        else:
            if self.nFree/self.nUnknown > Hex.Tr:
                return 0
            else:
                return -1


class FractionalHex(Hex):
    """
    A subclass of decentralized_exploration.helpers.hex_grid.Hex with float axial coordinates. 
        Used to convert to nearest integer hex

    Public Methods
    ----------
    to_hex(): returns decentralized_exploration.helpers.hex_grid.Hex object that is 
        nearest in axial coordinates
    """

    def to_hex(self):
        q = round(self.q)
        r = round(self.r)
        s = round(self.s)
        dq = abs(q - self.q)
        dr = abs(r - self.r)
        ds = abs(s - self.s)
        if(dq > dr and dq > ds):
            q = -(r + s)
        elif(dr > ds):
            r = -(q + s)
        return Hex(int(q), int(r), self.node_id, self.nUnknown, self.nFree, self.nOccupied)


class Orientation():
    """
    A class used to define different orientations of hex grids, such as flat on top or pointy on top

    Instance Attributes
    ----------
    f(list): list of floats representing flatted conversion matrix
    b(list): list of floats representing flatted inverse conversion matrix
    start_angle(float): the starting angle before rotation
    """

    def __init__(self, f, b, start_angle):
        self.f = f
        self.b = b
        self.start_angle = start_angle


OrientationPointy = Orientation(
    f=[math.sqrt(3.0), math.sqrt(3.0)/2.0, 0.0, 3.0/2.0],
    b=[math.sqrt(3.0)/3.0, -1.0/3.0, 0.0, 2.0/3.0],
    start_angle=0.5)

OrientationFlat = Orientation(
    f=[3.0/2.0, 0.0, math.sqrt(3.0)/2.0, math.sqrt(3.0)],
    b=[2.0/3.0, 0.0, -1.0/3.0, math.sqrt(3.0)/3.0],
    start_angle=0.0)


class Grid():
    """
    A class used to represent a hexagonal grid (represented as a graph)

    Instance Attributes
    ----------
    orientation (decentralized_exploration.helpers.hex_grid.Orientation): an Orientation object whether 
        the grid is flat-topped or pointy-topped
    origin (list) : a 2-element list of coordinates for the origin of the grid
    size (float) : the size of the hexagons
    graph(networkx.classes.graph.Graph): a networkx Graph object 
    allHexes(list): the list of all decentralized_exploration.helpers.hex_grid.Hex objects in the graph

    Public Methods
    -------
    find_hex(desired_hex): if there is a hex in allHexes with the given axial coordinates, returns it. 
        Otherwise, returns None
    add_hex(new_hex): adds a given Hex object to graph. If that Hex already exists, does nothing. Returns node_id
    hex_at(point): returns Hex with axial coordinates of Hex covering given pixel coordinate
    has_unexplored(): returns true if there are unexplored Hexs in allHexes
    hex_neighbours(center_hex): returns list of node_ids of adjacent neighbours of given Hex
    update_hex(node_id, nUnknown = 0, nFree = 0, nOccupied = 0): updates the number of unknown, free, 
        and occupied pixels of a given Hex
    hex_center(hexagon): returns the sub-pixel coordinates of the center of a given Hex
    """

    def __init__(self, orientation, origin, size):
        self.orientation = orientation
        self.origin = origin
        self.size = size
        self.graph = nx.Graph()
    
    @property
    def allHexes(self):
        return [self.graph.nodes()[node]['hex'] for node in list(self.graph.nodes())] 

    # Public Methods
    def find_hex(self, desired_hex):
        """
        Finds and returns the Hex in the Grid with given coordinates if there is one

        Parameters
        ----------
        desired_hex (decentralized_exploration.helpers.hex_grid.Hex): a Hex object with desired axial coordinates

        Returns
        ----------
        decentralized_exploration.helpers.hex_grid.Hex: the desired hex in Grid, or None if there is none
        """

        found_hex = [h for h in self.allHexes if h.q == desired_hex.q and h.r == desired_hex.r]
        if len(found_hex) == 1:
            return found_hex[0]
        else:
            return None
    
    def add_hex(self, new_hex):
        """
        Adds a Hex with given axial coordinates into the graph. If there is already a Hex at 
            that position, simply returns that Hex's node_id. Otherwise, returns new node_id
            of the added Hex

        Parameters
        ----------
        new_hex (decentralized_exploration.helpers.hex_grid.Hex): a Hex object with axial coordinates

        Returns
        ----------
        int: the node_id of the added Hex
        """

        found_hex = self.find_hex(new_hex)
        
        if found_hex:
            return found_hex.node_id
        else:
            node_id = len(self.graph.nodes())
            new_hex.node_id = node_id
            self.graph.add_node(node_id, hex = new_hex)

            if new_hex.state == 0:
                neighbours = self.hex_neighbours(new_hex)

                for neighbour in neighbours:
                    if self.graph.nodes[neighbour]['hex'].state == 0:
                        self.graph.add_edge(node_id, neighbour)

            return node_id

    def hex_at(self, point):
        """
        Returns Hex with axial coordinates that covers the specified pixel point

        Parameters
        ----------
        point (list): a 2-element list of pixel coordinates

        Returns
        ----------
        decentralized_exploration.helpers.hex_grid.Hex: the Hex axial coordinates covering that point
        """

        x = (point[1] - self.origin[0]) / float(self.size)
        y = (point[0] - self.origin[1]) / float(self.size)
        q = self.orientation.b[0]*x + self.orientation.b[1] * y
        r = self.orientation.b[2]*x + self.orientation.b[3] * y

        return FractionalHex(q, r).to_hex()

    def has_unexplored(self):
        """
        Checks whether there are hexes left to explore 

        Returns
        ----------
        boolean: True if there are unexplored hexes in allHexes, False otherwise
        """

        unexplored_hexes = sum([h.state == -1 for h in self.allHexes])
        if unexplored_hexes > 0:
            return True
        else:
            return False
    
    def hex_neighbours(self, center_hex):
        """
        Returns list of directly adjacent neighbours of a given Hex

        Parameters
        ----------
        center_hex (decentralized_exploration.helpers.hex_grid.Hex): a Hex object

        Returns
        ----------
        list: a list of node_ids for every hex neighbour
        """

        neighbours = []

        for q in range(-1, 2):
            for r in range(-1, +2):
                if q != r:
                    neighbour = self.find_hex(Hex(center_hex.q + q, center_hex.r + r))
                    if neighbour:
                        neighbours.append(neighbour.node_id)
        
        return neighbours
    
    def update_hex(self, node_id, dUnknown = 0, dFree = 0, dOccupied = 0):
        """
        Updates a Hex with changes to the number of unknown, free, and occupied pixels

        Parameters
        ----------
        node_id (int): the node_id of the Hex to be updated
        dUnknown (int): the difference in unknown pixels 
        dFree (int): the difference in free pixels 
        dOccupied (int): the difference in occupied pixels 
        """

        old_hex = self.graph.nodes[node_id]['hex']
        old_state = old_hex.state

        old_hex.nUnknown += dUnknown
        old_hex.nFree += dFree
        old_hex.nOccupied += dOccupied

        new_state = old_hex.state

        if old_state != 0 and new_state == 0:
            neighbours = self.hex_neighbours(old_hex)

            for neighbour in neighbours:
                if self.graph.nodes[neighbour]['hex'].state == 0:
                    self.graph.add_edge(node_id, neighbour)
        
        elif old_state == 0 and new_state != 0:
            neighbours = list(self.graph.neighbors(node_id))
            for neighbour in neighbours:
                self.graph.remove_edge(node_id, neighbour)
    
    def hex_center(self, hexagon):
        """
        Returns the sub-pixel coordinates of the center of the hexagon

        Parameters
        ----------
        hexagon (decentralized_exploration.helpers.hex_grid.Hex): the Hex with axial coordinates

        Returns
        ----------
        list: a 2-element array of sub-pixel coordinates
        """

        f = self.orientation.f
        x = (f[0] * hexagon.q + f[1]*hexagon.r)*self.size + self.origin[0]
        y = (f[2] * hexagon.q + f[3]*hexagon.r)*self.size + self.origin[1]
        return [y, x]


def convert_pixelmap_to_grid(pixel_map, size):
    """
    Converts an image (represented as a numpy.ndarray) into a grid

    Parameters
    ----------
    pixel_map (numpy.ndarry): numpy array of pixels representing the map. 
        -1 == unexplored
        0  == free
        1  == occupied
    size (float): the size of the hexagons compared to the pixels in pixel_map

    Returns
    ----------
    decentralized_exploration.helpers.hex_grid.Grid: a Grid object representing the map
    """

    center = [0, 0]
    grid = Grid(OrientationFlat, center, size)

    for y in range(pixel_map.shape[0]):
        for x in range(pixel_map.shape[1]):
            found_hex = grid.hex_at([y, x])

            node_id = grid.add_hex(found_hex)
            
            if pixel_map[y][x] == 0:
                grid.update_hex(node_id = node_id, dFree = 1)
            elif pixel_map[y][x] == 1:
                grid.update_hex(node_id = node_id, dOccupied = 1)   
            elif pixel_map[y][x] == -1:
                grid.update_hex(node_id = node_id, dUnknown = 1)         

    return grid