import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx


def plot_grid(grid, plot, robot_pos=[], robot_orientation=0):
    """
    Plots a given Grid. If a robot_pos is given, will highlight the hexagon the robot is in in red

    Parameters
    ----------
    grid (decentralized_exploration.helpers.hex_grid.Grid): the grid to be plotted
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    robot_pos (list, optional): an optional 2-element array of pixel coordinates
    """

    plot.cla()

    all_hexes = grid.all_hexes
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in all_hexes]
    colors = [colors_list[h.state+1] for h in all_hexes]
    labels = [h.node_id for h in all_hexes]

    rewards = {}
    for node in all_hexes:
        if node.reward > 0:
            y = 2. * np.sin(np.radians(60)) * (node.r - node.s) / 3.
            rewards[(node.q, y)] = node.reward

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    plot.set_aspect('equal')

    rx, ry = float('inf'), float('inf')
    if len(robot_pos) == 2:
        robot_hex = grid.hex_at(robot_pos)
        rx = robot_hex.q
        ry = 2.*np.sin(np.radians(60)) * (robot_hex.r - robot_hex.s)/3.

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        alpha = 0.5
        
        if (x, y) in rewards:
            c = 'blue'
            alpha = 1/(1+np.exp(-rewards[(x, y)]/7))
            plot.text(x, y+0.2, rewards[(x, y)], ha='center', va='center', size=5)

        if rx == x and ry == y:
            c = 'red'
            alpha = 0.5

            if robot_orientation == 1:
                plot.plot(x, y-0.3, 'mo')
            if robot_orientation == 2:
                plot.plot(x-0.2, y-0.2, 'mo')
            if robot_orientation == 3:
                plot.plot(x-0.2, y+0.2, 'mo')
            if robot_orientation == 4:
                plot.plot(x, y+0.3, 'mo')
            if robot_orientation == 5:
                plot.plot(x+0.2, y+0.2, 'mo')
            if robot_orientation == 6:
                plot.plot(x+0.2, y-0.2, 'mo')


        hexagon = RegularPolygon((x, y), numVertices=6, radius=2./3.,
                                 orientation=np.radians(30),
                                 facecolor=c, alpha=alpha, edgecolor='k')
        plot.add_patch(hexagon)
        plot.text(x, y-0.2, l, ha='center', va='center', size=5)

    plot.set_xlim([min(hcoord)-1, max(hcoord)+1])
    plot.set_ylim([min(vcoord)-1, max(vcoord)+1])
    plot.invert_yaxis()


def plot_path(grid, start_node, end_node, plot):
    """
    Plots a given Grid as well as the path from a starting Hex to an end Hex in red

    Parameters
    ----------
    grid (decentralized_exploration.helpers.hex_grid.Grid): the grid to be plotted
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    start_node (int): the node_id of the starting Hex
    end_node (int): the node_id of the end Hex
    """
    
    plot.cla()

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

    plot.set_aspect('equal')

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        if l in path:
            c = 'red'

        hexagon = RegularPolygon((x, y), numVertices=6, radius=2./3.,
                                 orientation=np.radians(30),
                                 facecolor=c, alpha=0.5, edgecolor='k')
        plot.add_patch(hexagon)
        plot.text(x, y-0.2, l, ha='center', va='center', size=5)

    plot.set_xlim([min(hcoord)-1, max(hcoord)+1])
    plot.set_ylim([min(vcoord)-1, max(vcoord)+1])
    plot.invert_yaxis()
    return plot


def plot_map(pixel_map, plot, robot_pos=[]):
    """
    Converts an image (represented as a numpy.ndarray) into a grid

    Parameters
    ----------
    pixel_map (numpy.ndarry): numpy array of pixels representing the map.
        -1 == unexplored
        0  == free
        1  == occupied
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    robot_pos (list, optional): an optional 2-element array of pixel coordinates
    """

    plot.cla()

    shaded_map = -pixel_map - (pixel_map == -1).astype(int)*1.5
    plot.imshow(shaded_map, cmap='gray')

    if len(robot_pos) == 2:
        plot.plot(robot_pos[1], robot_pos[0], 'ro')
