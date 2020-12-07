import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx

def plot_grid(grid, robot_pos = []):
    allHexes = grid.allHexes
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in allHexes]
    colors = [colors_list[h.state+1] for h in allHexes]
    labels = [h.node_id for h in allHexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    rx, ry = float('inf'), float('inf')
    if len(robot_pos) == 2:
        robot_hex = grid.hex_at(robot_pos)
        rx = robot_hex.q
        ry = 2.*np.sin(np.radians(60)) * (robot_hex.r - robot_hex.s)/3.


    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        if rx == x and ry == y:
            c = 'red'

        hex = RegularPolygon((x, y), numVertices=6, radius=2./3., 
                            orientation=np.radians(30), 
                            facecolor=c, alpha=0.5, edgecolor='k')
        ax.add_patch(hex)
        ax.text(x, y-0.2, l, ha='center', va='center', size=5)

    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.gca().invert_yaxis()
    plt.show()


def plot_path(grid, start, end):
    path = nx.shortest_path(grid.graph, start, end)

    allHexes = grid.allHexes
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in allHexes]
    colors = [colors_list[h.state+1] for h in allHexes]
    labels = [h.node_id for h in allHexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        if l in path:
            c = 'red'

        hex = RegularPolygon((x, y), numVertices=6, radius=2./3., 
                            orientation=np.radians(30), 
                            facecolor=c, alpha=0.5, edgecolor='k')
        ax.add_patch(hex)
        ax.text(x, y+0.2, l, ha='center', va='center', size=5)

    # Also add scatter points in hexagon centres
    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.gca().invert_yaxis()
    plt.show()


def plot_map(I, robot_pos = []):
    shaded_I = -I - (I == -1).astype(int)*1.5
    plt.imshow(shaded_I, cmap = 'gray')

    if len(robot_pos) == 2:
        plt.plot(robot_pos[1], robot_pos[0], 'ro')

    plt.show()