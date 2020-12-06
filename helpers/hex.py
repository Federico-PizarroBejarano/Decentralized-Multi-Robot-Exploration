import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Hex:
    # Tunable Parameter
    Tr = 0.5

    # Starting value for nOccupied should not be 0
    def __init__(self, q, r, nUnknown = 0, nFree = 0, nOccupied = 0):
        self.q = q
        self.r = r
        self.nUnknown = nUnknown
        self.nFree = nFree
        self.nOccupied = nOccupied

    @property
    def s(self):
        return -(self.q + self.r)
    
    def state(self):
        if self.nOccupied > 0:
            return 1
        elif self.nUnknown == 0:
            return 0
        else:
            if self.nFree/self.nUnknown > Tr:
                return 0
            else:
                return -1


class FractionalHex(Hex):
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
        return Hex(int(q), int(r), self.nUnknown, self.nFree, self.nOccupied)


class Orientation():
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
    def __init__(self, orientation, origin, size, allHexes = []):
        self.orientation = orientation
        self.origin = origin
        self.size = size
        self.allHexes = allHexes

    def find_hex(self, desired_hex):
        found_hex = [h for h in self.allHexes if h.q == desired_hex.q and h.r == desired_hex.r]
        if len(found_hex) == 1:
            return found_hex[0]
        elif len(found_hex) > 1:
            raise ValueError('More than 1 hex at spot q:%d r:%d', q, r)
        else:
            return False
    
    def add_hex(self, new_hex):
        found_hex = self.find_hex(new_hex)
        
        if found_hex:
            return False
        else:
            self.allHexes.append(new_hex)
            return True

    def hex_at(self, point):
        # type: (Point) -> Hex
        x = (point.x - self.origin.x) / float(self.size.x)
        y = (point.y - self.origin.y) / float(self.size.y)
        q = self.orientation.b[0]*x + self.orientation.b[1] * y
        r = self.orientation.b[2]*x + self.orientation.b[3] * y

        hex_position = FractionalHex(q, r).to_hex()
        found_hex = self.find_hex(hex_position)

        if found_hex:
            return True, found_hex
        else:
            return False, hex_position

    def hex_center(self, hex):
        # type: (Hex) -> Point
        f = self.orientation.f
        x = (f[0] * hex.q + f[1]*hex.r)*self.size.x + self.origin.x
        y = (f[2] * hex.q + f[3]*hex.r)*self.size.y + self.origin.y
        return Point(x, y)

    def has_unexplored(self):
        unexplored_hexes = [h for h in self.allHexes if h.state == -1]
        if len(unexplored_hexes) > 0:
            return True
        else:
            return False

def convert_image_to_grid(I, size = 8):
    center = Point(0, 0)
    size = Point(size, size)

    grid = Grid(OrientationFlat, center, size)

    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            found, found_hex = grid.hex_at(Point(x, y))

            if I[y][x] == 0:
                found_hex.nFree += 1
            elif I[y][x] == 1:
                found_hex.nOccupied += 1

            if not found:
                grid.add_hex(found_hex)

    return grid


def plot_grid(grid):
    allHexes = grid.allHexes
    colors_list = ['white', 'black']
    coord = [[h.q, h.r, h.s] for h in allHexes]
    colors = [colors_list[h.state()] for h in allHexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c in zip(hcoord, vcoord, colors):
        color = c[0]
        hex = RegularPolygon((x, y), numVertices=6, radius=2./3., 
                            orientation=np.radians(30), 
                            facecolor=color, alpha=0.2, edgecolor='k')
        ax.add_patch(hex)

    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, c=[c[0].lower() for c in colors], alpha=0.5)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    I = np.load('./maps/map_1.npy')
    grid = convert_image_to_grid(I, 10)
    plot_grid(grid)