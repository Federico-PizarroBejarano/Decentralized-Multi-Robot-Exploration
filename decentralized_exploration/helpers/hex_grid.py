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
    reward(int): the reward associated with this hex
    state(int): the state of this hexagon as unknown (-1), free (0), or occupied (1)
    """

    # Tunable Parameter
    Tr = 0.5

    # Starting value for nOccupied should not be 0
    def __init__(self, q, r, node_id=-1, nUnknown=0, nFree=0, nOccupied=0, reward=0):
        self.q = q
        self.r = r
        self.node_id = node_id
        self.nUnknown = nUnknown
        self.nFree = nFree
        self.nOccupied = nOccupied
        self.reward = reward

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
    A subclass of Hex with float axial coordinates. 
        Used to convert to nearest integer hex

    Public Methods
    ----------
    to_hex(): returns Hex object that is 
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
        return Hex(int(q), int(r), self.node_id, self.nUnknown, self.nFree, self.nOccupied, self.reward)


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
    orientation (Orientation): an Orientation object whether 
        the grid is flat-topped or pointy-topped
    origin (list) : a 2-element list of coordinates for the origin of the grid
    size (float) : the size of the hexagons
    graph(networkx.classes.graph.Graph): a networkx Graph object 
    all_hexes(list): the list of all Hex objects in the graph

    Public Methods
    -------
    find_hex(desired_hex): if there is a hex in all_hexes with the given axial coordinates, returns it. 
        Otherwise, returns None
    add_hex(new_hex): adds a given Hex object to graph. If that Hex already exists, does nothing. Returns node_id
    hex_at(point): returns Hex with axial coordinates of Hex covering given pixel coordinate
    has_unexplored(): returns true if there are unexplored Hexs in all_hexes
    hex_neighbours(center_hex): returns list of node_ids of adjacent neighbours of given Hex
    update_hex(node_id, nUnknown = 0, nFree = 0, nOccupied = 0): updates the number of unknown, free, 
        and occupied pixels of a given Hex
    hex_center(hexagon): returns the sub-pixel coordinates of the center of a given Hex
    """

    # Tunable Parameter
    radius = 2

    def __init__(self, orientation, origin, size):
        self.orientation = orientation
        self.origin = origin
        self.size = size
        self.graph = nx.Graph()

    @property
    def all_nodes(self):
        return self.graph.nodes()

    @property
    def all_hexes(self):
        return [self.all_nodes[node]['hex'] for node in list(self.all_nodes)]

    # Public Methods
    def find_hex(self, desired_hex):
        """
        Finds and returns the Hex in the Grid with given coordinates if there is one

        Parameters
        ----------
        desired_hex (Hex): a Hex object with desired axial coordinates

        Returns
        ----------
        Hex: the desired hex in Grid, or None if there is none
        """

        found_hex = [h for h in self.all_hexes if h.q == desired_hex.q and h.r == desired_hex.r]
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
        new_hex (Hex): a Hex object with axial coordinates

        Returns
        ----------
        int: the node_id of the added Hex
        """

        found_hex = self.find_hex(new_hex)

        if found_hex:
            return found_hex.node_id
        else:
            node_id = len(self.all_nodes)
            new_hex.node_id = node_id
            self.graph.add_node(node_id, hex=new_hex)

            if new_hex.state == 0:
                neighbours = self.hex_neighbours(new_hex)

                for neighbour in neighbours:
                    if self.all_nodes[neighbour]['hex'].state == 0:
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
        Hex: the Hex axial coordinates covering that point
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
        boolean: True if there are unexplored hexes in all_hexes, False otherwise
        """

        for h in self.all_hexes:
            if h.state == -1:
                return True
        
        return False

    def hex_neighbours(self, center_hex, radius=1):
        """
        Returns list of directly adjacent neighbours of a given Hex

        Parameters
        ----------
        center_hex (Hex): a Hex object

        Returns
        ----------
        list: a list of node_ids for every hex neighbour
        """

        neighbours = []

        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if (abs(q+r) <= radius) and ((q != r) or (q <= radius - 1 and q != 0)):
                    neighbour = self.find_hex(Hex(center_hex.q + q, center_hex.r + r))
                    if neighbour:
                        neighbours.append(neighbour.node_id)

        return neighbours

    def update_hex(self, node_id, dUnknown=0, dFree=0, dOccupied=0):
        """
        Updates a Hex with changes to the number of unknown, free, and occupied pixels

        Parameters
        ----------
        node_id (int): the node_id of the Hex to be updated
        dUnknown (int): the difference in unknown pixels 
        dFree (int): the difference in free pixels 
        dOccupied (int): the difference in occupied pixels 
        """

        old_hex = self.all_nodes[node_id]['hex']
        old_state = old_hex.state

        old_hex.nUnknown += dUnknown
        old_hex.nFree += dFree
        old_hex.nOccupied += dOccupied

        new_state = old_hex.state

        if old_state != 0 and new_state == 0:
            neighbours = self.hex_neighbours(old_hex)

            for neighbour in neighbours:
                if self.all_nodes[neighbour]['hex'].state == 0:
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
        hexagon (Hex): the Hex with axial coordinates

        Returns
        ----------
        list: a 2-element array of sub-pixel coordinates
        """

        f = self.orientation.f
        x = (f[0] * hexagon.q + f[1]*hexagon.r)*self.size + self.origin[0]
        y = (f[2] * hexagon.q + f[3]*hexagon.r)*self.size + self.origin[1]
        return [y, x]
    
    def propagate_rewards(self):
        """
        Clears the reward from all hexes and then re-calculates the reward  at every
        hex. Does not accept any arguments and does not return anything
        """

        for node in self.all_hexes:
            node.reward = 0
        
        for node in self.all_hexes:
            if node.state == -1:
                neighbours = self.hex_neighbours(node, self.radius)

                for neighbour in neighbours:
                    if self.all_nodes[neighbour]['hex'].state == 0 and self.clear_path(node, self.all_nodes[neighbour]['hex']):
                        self.all_nodes[neighbour]['hex'].reward += 1
    
    def hex_distance(self, start_hex, end_hex):
        """
        Takes two hexes and returns the hex distance between them as an integer

        Parameters
        ----------
        start_hex (Hex): a Hex object representing the starting hex
        end_hex (Hex): a Hex object representing the ending hex

        Returns
        ----------
        distance (int): a integer representing the Hex distance between two hexes
        """

        s_q, s_r = start_hex.q, start_hex.r
        e_q, e_r = end_hex.q, end_hex.r

        distance = (abs(s_q - e_q) + abs(s_q + s_r - e_q - e_r) + abs(s_r - e_r)) / 2
        return int(distance)

    def clear_path(self, start_hex, end_hex):
        """
        Determines if the direct linear path between two hexes is completely clear (all free hexes)

        Parameters
        ----------
        start_hex (Hex): a Hex object representing the starting hex
        end_hex (Hex): a Hex object representing the ending hex

        Returns
        ----------
        clear (bool): True if clear, False otherwise
        """

        distance = self.hex_distance(start_hex, end_hex)
        s_x, s_y = self.hex_center(start_hex)
        e_x, e_y = self.hex_center(end_hex)

        for i in range(1, distance):
            x = s_x + (e_x - s_x) * (1/distance) * i
            y = s_y + (e_y - s_y) * (1/distance) * i

            intermediate_hex = self.hex_at([x, y])
            intermediate_hex = self.find_hex(intermediate_hex)

            if not intermediate_hex or intermediate_hex.state != 0:
                return False
        
        return True
    
    def find_closest_unknown(self, center_hex):
        """
        If the given hex has a reward, returns one of the unknown hexes providing the reward

        Parameters
        ----------
        center_hex (Hex): a Hex object representing the ending hex

        Returns
        ----------
        unknown_hex (Hex): a Hex representing the neighbouring unknown hex
        """

        if center_hex.reward == 0:
            return None
        else:
            neighbours = self.hex_neighbours(center_hex, self.radius)

            for neighbour in neighbours:
                if self.all_nodes[neighbour]['hex'].state == -1 and self.clear_path(center_hex, self.all_nodes[neighbour]['hex']):
                    return neighbour



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
    Grid: a Grid object representing the map
    """

    center = [0, 0]
    grid = Grid(OrientationFlat, center, size)

    for y in range(pixel_map.shape[0]):
        for x in range(pixel_map.shape[1]):
            found_hex = grid.hex_at([y, x])

            node_id = grid.add_hex(found_hex)

            if pixel_map[y][x] == 0:
                grid.update_hex(node_id=node_id, dFree=1)
            elif pixel_map[y][x] == 1:
                grid.update_hex(node_id=node_id, dOccupied=1)
            elif pixel_map[y][x] == -1:
                grid.update_hex(node_id=node_id, dUnknown=1)

    return grid
