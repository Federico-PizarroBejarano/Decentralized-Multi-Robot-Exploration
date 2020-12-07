import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx


def plot_grid(grid, robot_pos=[]):
    """
    Plots a given Grid. If a robot_pos is given, will highlight the hexagon the robot is in in red

    Parameters
    ----------
    grid (decentralized_exploration.helpers.hex_grid.Grid): the grid to be plotted
    robot_pos (list, optional): an optional 2-element array of pixel coordinates
    """

    all_hexes = grid.all_hexes
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in all_hexes]
    colors = [colors_list[h.state+1] for h in all_hexes]
    labels = [h.node_id for h in all_hexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    fig = plt.figure()
    ax = fig.add_subplot(111)
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

        hexagon = RegularPolygon((x, y), numVertices=6, radius=2./3.,
                                 orientation=np.radians(30),
                                 facecolor=c, alpha=0.5, edgecolor='k')
        ax.add_patch(hexagon)
        ax.text(x, y-0.2, l, ha='center', va='center', size=5)

    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.gca().invert_yaxis()
    plt.show()


def plot_path(grid, start_node, end_node):
    """
    Plots a given Grid as well as the path from a starting Hex to an end Hex in red

    Parameters
    ----------
    grid (decentralized_exploration.helpers.hex_grid.Grid): the grid to be plotted
    start_node (int): the node_id of the starting Hex
    end_node (int): the node_id of the end Hex
    """

    # pylint:disable-msg=too-many-function-args
    path = nx.shortest_path(grid.graph, start_node, end_node)

    all_hexes = grid.all_hexes
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in all_hexes]
    colors = [colors_list[h.state+1] for h in all_hexes]
    labels = [h.node_id for h in all_hexes]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        if l in path:
            c = 'red'

        hexagon = RegularPolygon((x, y), numVertices=6, radius=2./3.,
                                 orientation=np.radians(30),
                                 facecolor=c, alpha=0.5, edgecolor='k')
        ax.add_patch(hexagon)
        ax.text(x, y+0.2, l, ha='center', va='center', size=5)

    plt.axis([min(hcoord)-1, max(hcoord)+1, min(vcoord)-1, max(vcoord)+1])
    plt.gca().invert_yaxis()
    plt.show()


def plot_map(pixel_map, robot_pos=[]):
    """
    Converts an image (represented as a numpy.ndarray) into a grid

    Parameters
    ----------
    pixel_map (numpy.ndarry): numpy array of pixels representing the map.
        -1 == unexplored
        0  == free
        1  == occupied
    robot_pos (list, optional): an optional 2-element array of pixel coordinates
    """

    shaded_map = -pixel_map - (pixel_map == -1).astype(int)*1.5
    plt.imshow(shaded_map, cmap='gray')

    if len(robot_pos) == 2:
        plt.plot(robot_pos[1], robot_pos[0], 'ro')

    plt.show()
