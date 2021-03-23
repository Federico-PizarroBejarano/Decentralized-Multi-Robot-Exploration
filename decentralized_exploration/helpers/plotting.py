import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


def plot_grid(grid, plot, robot_states = {}, mode='value'):
    """
    Plots a given Grid. If a robot_pos is given, will highlight the hexagon the robot is in in red

    Parameters
    ----------
    grid (Grid): the grid to be plotted
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    robot_states (list): an optional dictionary where the keys are the robot_ids and the values are RobotStates
    mode (str) = either 'value' to show the value of each hex, 'reward' to show the reward at each hex, or neither to show nothing
    """

    plot.cla()

    all_hexes = grid.all_hexes.values()
    colors_list = ['0.5', '1', '0']
    coord = [[h.q, h.r, h.s] for h in all_hexes]
    colors = [colors_list[h.state+1] for h in all_hexes]

    rewards = {}
    max_reward = -float('inf')
    min_reward = float('inf')

    if mode == 'value':
        for hexagon in all_hexes:
            if hexagon.V != 0 and hexagon.state == 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.V, 1)
                if round(hexagon.V, 1) > max_reward:
                    max_reward = round(hexagon.V, 1)
                if round(hexagon.V, 1) < min_reward:
                    min_reward = round(hexagon.V, 1)
    elif mode == 'reward':
        for hexagon in all_hexes:
            if hexagon.reward != 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.reward, 1)
                if round(hexagon.reward, 1) > max_reward:
                    max_reward = round(hexagon.reward, 1)
    if mode == 'probability':
        for hexagon in all_hexes:
            if hexagon.probability != 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.probability*100, 2)
                if round(hexagon.probability*100, 2) > max_reward:
                    max_reward = round(hexagon.probability*100, 2)

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    plot.set_aspect('equal')

    hex_robot_states = {}
    for robot_state in robot_states.values():
        robot_hex = grid.hex_at(point=robot_state.pixel_position)
        hex_x = robot_hex.q
        hex_y = 2.*np.sin(np.radians(60)) * (robot_hex.r - robot_hex.s)/3.

        hex_robot_states[(hex_x, hex_y)] = robot_state.orientation

    # Add some coloured hexagons
    for x, y, c in zip(hcoord, vcoord, colors):  
        alpha = 0.5      
        if (x, y) in rewards:
            plot.text(x, y, rewards[(x, y)], ha='center', va='center', size=8)
            if rewards[(x, y)] > 0:
                c = 'green'
                alpha = rewards[(x, y)]/max_reward
            elif rewards[(x, y)] < 0:
                c = 'red'
                alpha = rewards[(x, y)]/min_reward

        if (x, y) in hex_robot_states:
            alpha = 0.5
            c = 'yellow'

            if hex_robot_states[(x, y)] == 1:
                plot.plot(x, y-0.3, 'bo')
            if hex_robot_states[(x, y)] == 2:
                plot.plot(x-0.25, y-0.15, 'bo')
            if hex_robot_states[(x, y)] == 3:
                plot.plot(x-0.25, y+0.15, 'bo')
            if hex_robot_states[(x, y)] == 4:
                plot.plot(x, y+0.3, 'bo')
            if hex_robot_states[(x, y)] == 5:
                plot.plot(x+0.25, y+0.15, 'bo')
            if hex_robot_states[(x, y)] == 6:
                plot.plot(x+0.25, y-0.15, 'bo')


        hexagon = RegularPolygon((x, y), numVertices=6, radius=2./3.,
                                 orientation=np.radians(30),
                                 facecolor=c, alpha=alpha, edgecolor='k')
        plot.add_patch(hexagon)

    plot.set_xlim([min(hcoord)-1, max(hcoord)+1])
    plot.set_ylim([min(vcoord)-1, max(vcoord)+1])
    plot.invert_yaxis()


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
    robot_pos (list): an optional 2-element array of pixel coordinates
    """

    plot.cla()

    shaded_map = -pixel_map - (pixel_map == -1).astype(int)*1.5
    plot.imshow(shaded_map, cmap='gray')

    if len(robot_pos) == 2:
        plot.plot(robot_pos[1], robot_pos[0], 'ro')
