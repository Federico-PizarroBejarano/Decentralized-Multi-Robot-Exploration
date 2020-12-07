import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import networkx as nx

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Hex:
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
    def __init__(self, orientation, origin, size):
        self.orientation = orientation
        self.origin = origin
        self.size = size
        self.graph = nx.Graph()
    
    @property
    def allHexes(self):
        return [self.graph.nodes()[node]['hex'] for node in list(self.graph.nodes())] 

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
            return found_hex.node_id
        else:
            node_id = len(self.graph.nodes())
            new_hex.node_id = node_id
            self.graph.add_node(node_id, hex = new_hex)

            if new_hex.state == 0:
                neighbours = self.hex_neighbours(new_hex)

                for neighbour in neighbours:
                    if neighbour[1] == 0:
                        self.graph.add_edge(node_id, neighbour[0])

            return node_id

    def hex_at(self, point):
        x = (point.x - self.origin.x) / float(self.size.x)
        y = (point.y - self.origin.y) / float(self.size.y)
        q = self.orientation.b[0]*x + self.orientation.b[1] * y
        r = self.orientation.b[2]*x + self.orientation.b[3] * y

        return FractionalHex(q, r).to_hex()

    def has_unexplored(self):
        unexplored_hexes = sum([h.state == -1 for h in self.allHexes])
        if unexplored_hexes > 0:
            return True
        else:
            return False
    
    def hex_neighbours(self, center_hex):
        neighbours = []

        for q in range(-1, 2):
            for r in range(-1, +2):
                if not (q==0 and r==0):
                    neighbour = self.find_hex(Hex(center_hex.q + q, center_hex.r + r))
                    if neighbour:
                        neighbours.append([neighbour.node_id, neighbour.state])
        
        return neighbours
    
    def update_hex(self, node_id, nUnknown = 0, nFree = 0, nOccupied = 0):
        old_hex = self.graph.nodes[node_id]['hex']
        old_state = old_hex.state

        old_hex.nUnknown += nUnknown
        old_hex.nFree += nFree
        old_hex.nOccupied += nOccupied

        new_state = old_hex.state

        if old_state != 0 and new_state == 0:
            neighbours = hex_neighbours(new_hex)

            for neighbour in neighbours:
                if neighbour[1] == 0:
                    self.graph.add_edge(node_id, neighbour[0])
        
        elif old_state == 0 and new_state != 0:
            neighbours = list(self.graph.neighbors(node_id))
            for neighbour in neighbours:
                self.graph.remove_edge(node_id, neighbour)
    
    def hex_center(self, hex):
        f = self.orientation.f
        x = (f[0] * hex.q + f[1]*hex.r)*self.size.x + self.origin.x
        y = (f[2] * hex.q + f[3]*hex.r)*self.size.y + self.origin.y
        return Point(x, y)


def convert_image_to_grid(I, size):
    center = Point(0, 0)
    size = Point(size, size)

    grid = Grid(OrientationFlat, center, size)

    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            found_hex = grid.hex_at(Point(x, y))

            node_id = grid.add_hex(found_hex)
            
            if I[y][x] == 0:
                grid.update_hex(node_id, nFree = 1)
            elif I[y][x] == 1:
                grid.update_hex(node_id, nOccupied = 1)   
            elif I[y][x] == -1:
                grid.update_hex(node_id, nUnknown = 1)         

    return grid


def plot_grid(grid):
    allHexes = grid.allHexes
    colors_list = ['white', 'black']
    coord = [[h.q, h.r, h.s] for h in allHexes]
    colors = [colors_list[h.state] for h in allHexes]

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


    plt.gca().invert_yaxis()
    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.show()

def plot_path(grid, start, end):
    path = nx.shortest_path(grid.graph, source = start, target=end)

    allHexes = grid.allHexes
    colors_list = ['white', 'black']
    coord = [[h.q, h.r, h.s] for h in allHexes]
    colors = [colors_list[h.state] for h in allHexes]
    labels = [h.node_id for h in allHexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        color = c[0]
        if l in path:
            color = 'red'

        hex = RegularPolygon((x, y), numVertices=6, radius=2./3., 
                            orientation=np.radians(30), 
                            facecolor=color, alpha=0.2, edgecolor='k')
        ax.add_patch(hex)
        ax.text(x, y+0.2, l, ha='center', va='center', size=5)

    # Also add scatter points in hexagon centres
    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    I = np.load('./maps/map_1_small.npy')
    grid = convert_image_to_grid(I, 7)
    plot_path(grid, 88, 25)