import math
import numpy as np

class Hex:
    """
    A class used to represent a hexagon in the hexagonal grid layer

    Class Attributes
    ----------------
    Tr (int): the ratio between free and unknown pixels required for the hexagon
        to be considered free. A tunable parameter

    Instance Attributes
    -------------------
    q (int): the first axial coordinate
    r (int): the second axial coordinate
    s (int): the third axial coordinate satisfying (q + r + s = 0)
    nUnknown (int): number of unknown pixels in this hexagon
    nFree (int): number of free pixels in this hexagon
    nOccupied (int): number of occupied pixels in this hexagon
    reward (int): the reward associated with this hex
    V (float): the true value of a Hex. Used for displaying computed values.
    state (int): the state of this hexagon as unknown (-1), free (0), or occupied (1)

    Public Methods
    --------------
    update_hex(nUnknown = 0, nFree = 0, nOccupied = 0): updates the number of unknown, free, 
        and occupied pixels
    """

    # Tunable Parameter
    Tr = 1.0

    # Starting value for nOccupied should not be 0
    def __init__(self, q, r, nUnknown=0.0, nFree=0.0, nOccupied=0.0, reward=0.0):
        self.q = q
        self.r = r
        self.nUnknown = nUnknown
        self.nFree = nFree
        self.nOccupied = nOccupied
        self.reward = reward
        self.V = 0.0
        self.probability = 0.0
        self.probability_steps = 0

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
            if float(self.nFree)/float(self.nUnknown) > Hex.Tr:
                return 0
            else:
                return -1
    
    # Public Methods
    def update_hex(self, dUnknown=0, dFree=0, dOccupied=0):
        """
        Updates the hex with changes to the number of unknown, free, and occupied pixels

        Parameters
        ----------
        dUnknown (int): the difference in unknown pixels 
        dFree (int): the difference in free pixels 
        dOccupied (int): the difference in occupied pixels 
        """

        self.nUnknown += dUnknown
        self.nFree += dFree
        self.nOccupied += dOccupied


class FractionalHex(Hex):
    """
    A subclass of Hex with float axial coordinates. 
        Used to convert to nearest integer hex

    Public Methods
    --------------
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
        return Hex(q=int(q), r=int(r))


class Orientation():
    """
    A class used to define different orientations of hex grids, such as flat on top or pointy on top

    Instance Attributes
    -------------------
    f (list): list of floats representing flatted conversion matrix
    b (list): list of floats representing flatted inverse conversion matrix
    start_angle (float): the starting angle before rotation
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
    A class used to represent a hexagonal grid (represented as an dictionary of Hex objects)

    Class Attributes
    ----------------
    radius (int): the radius for which rewards are propagated. A tunable parameter

    Instance Attributes
    -------------------
    orientation (Orientation): an Orientation object whether 
        the grid is flat-topped or pointy-topped
    origin (list) : a 2-element list of coordinates for the origin of the grid
    size (float) : the size of the hexagons
    all_hexes (dictionary): the dictionary of all Hex objects in the grid, indexed by (q, r)

    Public Methods
    --------------
    find_hex(desired_hex): if there is a hex in all_hexes with the given axial coordinates, returns it. 
        Otherwise, returns None
    add_hex(new_hex): adds a given Hex object to the grid. If that Hex already exists, does nothing. Returns Hex.
    hex_at(point): returns Hex with axial coordinates of Hex covering given pixel coordinate
    has_rewards(): returns True if there are Hexs with rewards in all_hexes
    hex_neighbours(center_hex): returns list of the adjacent neighbours of given Hex
    hex_center(hexagon): returns the sub-pixel coordinates of the center of a given Hex
    """

    # Tunable Parameter
    radius = 2

    def __init__(self, orientation, origin, size):
        self.orientation = orientation
        self.origin = origin
        self.size = size
        self.all_hexes = {}
    
    # Static Methods
    @staticmethod
    def hex_distance(start_hex, end_hex):
        """
        Takes two hexes and returns the hex distance between them as an integer

        Parameters
        ----------
        start_hex (Hex): a Hex object representing the starting hex
        end_hex (Hex): a Hex object representing the ending hex

        Returns
        -------
        distance (int): a integer representing the Hex distance between two hexes
        """

        s_q, s_r = start_hex.q, start_hex.r
        e_q, e_r = end_hex.q, end_hex.r

        distance = (abs(s_q - e_q) + abs(s_q + s_r - e_q - e_r) + abs(s_r - e_r)) / 2
        return int(distance)

    # Public Instance Methods
    def find_hex(self, desired_hex):
        """
        Finds and returns the Hex in the Grid with given coordinates if there is one

        Parameters
        ----------
        desired_hex (Hex): a Hex object with desired axial coordinates

        Returns
        -------
        found_hex (Hex): the desired hex in the grid, or None if there is none
        """

        if (desired_hex.q, desired_hex.r) in self.all_hexes:
            return self.all_hexes[(desired_hex.q, desired_hex.r)]
        else:
            return None

    def add_hex(self, new_hex):
        """
        Adds a Hex with given axial coordinates into all_hexes.

        Parameters
        ----------
        new_hex (Hex): a Hex object with axial coordinates

        Returns
        -------
        new_hex (Hex): either the found_hex (if found) or the passed in Hex object (if newly added)
        """

        found_hex = self.find_hex(desired_hex=new_hex)

        if not found_hex:
            self.all_hexes[(new_hex.q, new_hex.r)] = new_hex
            return new_hex
        else:
            return found_hex
        

    def hex_at(self, point):
        """
        Returns Hex with axial coordinates that covers the specified pixel point

        Parameters
        ----------
        point (list): a 2-element list of pixel coordinates

        Returns
        -------
        hex_at_point (Hex): the Hex axial coordinates covering that point
        """

        x = (point[1] - self.origin[0]) / float(self.size)
        y = (point[0] - self.origin[1]) / float(self.size)
        q = self.orientation.b[0]*x + self.orientation.b[1] * y
        r = self.orientation.b[2]*x + self.orientation.b[3] * y

        hex_at_point = FractionalHex(q=q, r=r).to_hex()
        return hex_at_point

    def has_rewards(self):
        """
        Checks whether there are hexes left to explore 

        Returns
        -------
        has_rewards (bool): True if there are hexes with rewadrs in all_hexes, False otherwise
        """

        for hexagon in self.all_hexes.values():
            if hexagon.reward > 0:
                return True
        
        return False

    def hex_neighbours(self, center_hex, radius=1):
        """
        Returns list of neighbours of a given Hex within a specified hex radius

        Parameters
        ----------
        center_hex (Hex): a Hex object
        radius (int): the radius to consider

        Returns
        -------
        neighbours (list): a list of Hex objects
        """

        neighbours = []

        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if (abs(q+r) <= radius) and ((q != r) or (q <= radius - 1 and q != 0)):
                    neighbour = self.find_hex(desired_hex=Hex(center_hex.q + q, center_hex.r + r))
                    if neighbour:
                        neighbours.append(neighbour)

        return neighbours

    def hex_center(self, hexagon):
        """
        Returns the sub-pixel coordinates of the center of the hexagon

        Parameters
        ----------
        hexagon (Hex): the Hex with axial coordinates

        Returns
        -------
        center_coords (list): a 2-element array of sub-pixel coordinates
        """

        f = self.orientation.f
        x = (f[0] * hexagon.q + f[1]*hexagon.r)*self.size + self.origin[0]
        y = (f[2] * hexagon.q + f[3]*hexagon.r)*self.size + self.origin[1]

        center_coords = [y, x]
        return center_coords
    
    def propagate_rewards(self):
        """
        Clears the reward from all hexes and then re-calculates the reward  at every
        hex. Does not accept any arguments and does not return anything
        """

        for hexagon in self.all_hexes.values():
            hexagon.reward = 0
        
        for hexagon in self.all_hexes.values():
            if hexagon.state == -1:
                neighbours = self.hex_neighbours(center_hex=hexagon, radius=self.radius)

                for neighbour in neighbours:
                    if neighbour.state == 0 and self.clear_path(start_hex=hexagon, end_hex=neighbour):
                        neighbour.reward += 1

    def clear_path(self, start_hex, end_hex):
        """
        Determines if the direct linear path between two hexes is completely clear (all free hexes)

        Parameters
        ----------
        start_hex (Hex): a Hex object representing the starting hex
        end_hex (Hex): a Hex object representing the ending hex

        Returns
        -------
        clear (bool): True if clear, False otherwise
        """

        distance = Grid.hex_distance(start_hex=start_hex, end_hex=end_hex)
        s_x, s_y = self.hex_center(hexagon=start_hex)
        e_x, e_y = self.hex_center(hexagon=end_hex)

        for i in range(1, distance):
            x = s_x + (e_x - s_x) * (1.0/distance) * i
            y = s_y + (e_y - s_y) * (1.0/distance) * i

            intermediate_hex = self.hex_at(point=[x, y])
            intermediate_hex = self.find_hex(desired_hex=intermediate_hex)

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
        -------
        unknown_hex (Hex): a Hex representing the neighbouring unknown hex
        """

        if center_hex.reward == 0:
            return None
        else:
            neighbours = self.hex_neighbours(center_hex=center_hex, radius=self.radius)

            for neighbour in neighbours:
                if neighbour.state == -1 and self.clear_path(start_hex=center_hex, end_hex=neighbour):
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
    -------
    grid (Grid): a Grid object representing the map
    """

    center = [0, 0]
    grid = Grid(orientation=OrientationFlat, origin=center, size=size)

    for y in range(pixel_map.shape[0]):
        for x in range(pixel_map.shape[1]):
            found_hex = grid.hex_at(point=[y, x])

            new_hex = grid.add_hex(new_hex=found_hex)

            if pixel_map[y][x] == 0:
                new_hex.update_hex(dFree=1)
            elif pixel_map[y][x] == 1:
                new_hex.update_hex(dOccupied=1)
            elif pixel_map[y][x] == -1:
                new_hex.update_hex(dUnknown=1)

    return grid
