import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import cPickle as pickle


def plot_grid(grid, plot, robot_states = {}, mode='value'):
    """
    Plots a given Grid. If a robot_pos is given, will highlight the hexagon the robot is in in red

    Parameters
    ----------
    grid (Grid): the grid to be plotted
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    robot_states (dict): an optional dictionary where the keys are the robot_ids and the values are RobotStates
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
    for robot in robot_states.keys():
        robot_hex = grid.hex_at(point=robot_states[robot].pixel_position)
        hex_x = robot_hex.q
        hex_y = 2.*np.sin(np.radians(60)) * (robot_hex.r - robot_hex.s)/3.

        hex_robot_states[(hex_x, hex_y)] = (robot_states[robot].orientation, robot)

    # Add some coloured hexagons
    for x, y, c in zip(hcoord, vcoord, colors):  
        alpha = 0.5      
        if (x, y) in rewards:
            # plot.text(x, y, rewards[(x, y)], ha='center', va='center', size=8)
            if rewards[(x, y)] > 0:
                c = 'green'
                alpha = rewards[(x, y)]/max_reward
            elif rewards[(x, y)] < 0:
                c = 'red'
                alpha = rewards[(x, y)]/min_reward

        if (x, y) in hex_robot_states:
            alpha = 0.8
            c = 'yellow'
            plot.text(x, y, hex_robot_states[(x, y)][1][-1], ha='center', va='center', size=8)

            if hex_robot_states[(x, y)][0] == 1:
                plot.plot(x, y-0.3, 'bo')
            if hex_robot_states[(x, y)][0] == 2:
                plot.plot(x-0.25, y-0.15, 'bo')
            if hex_robot_states[(x, y)][0] == 3:
                plot.plot(x-0.25, y+0.15, 'bo')
            if hex_robot_states[(x, y)][0] == 4:
                plot.plot(x, y+0.3, 'bo')
            if hex_robot_states[(x, y)][0] == 5:
                plot.plot(x+0.25, y+0.15, 'bo')
            if hex_robot_states[(x, y)][0] == 6:
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


def plot_one_set(results, plot=True):
    local_interactions = []
    to_75_pc = []
    to_90_pc = []
    total_iterations = []

    for test in results:
        num_of_local_interactions = np.sum(test[:, 1])
        
        iterations_to_90_pc = 0
        iterations_to_75_pc = 0

        for iteration in range(test.shape[0]):
            if test[iteration][0]*1773.0/(1773.0-83.0) > 0.90 and iterations_to_90_pc == 0:
                iterations_to_90_pc = iteration
            elif test[iteration][0]*1773.0/(1773.0-83.0) > 0.75 and iterations_to_75_pc == 0:
                iterations_to_75_pc = iteration
        
        local_interactions.append(num_of_local_interactions)
        to_75_pc.append(iterations_to_75_pc)
        to_90_pc.append(iterations_to_90_pc)
        total_iterations.append(test.shape[0])

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot('111')

        local_interactions, = ax.plot(range(1, len(results)+1), local_interactions, marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Cumulated iterations with local interactions')
        to_75_pc, = ax.plot(range(1, len(results)+1), to_75_pc, marker='o', linewidth=2, markersize=12, label='Iterations until 75% explored')
        to_90_pc, = ax.plot(range(1, len(results)+1), to_90_pc, marker='o', linewidth=2, markersize=12, label='Iterations until 90% explored')
        total_iterations, = ax.plot(range(1, len(results)+1), total_iterations, marker='o', linewidth=2, markersize=12, label='Iterations until 100% explored')

        plt.legend(handles=[local_interactions, to_75_pc, to_90_pc, total_iterations])
        ax.set_ylim(ymin=0)
        plt.show()
    else:
        cumulated_results = {}
        cumulated_results['local_interactions'] = sum(local_interactions)/len(results)
        cumulated_results['to_75_pc'] = sum(to_75_pc)/len(results)
        cumulated_results['to_90_pc'] = sum(to_90_pc)/len(results)
        cumulated_results['total_iterations'] = sum(total_iterations)/len(results)

        return cumulated_results


def plot_all_results():
    filenames = ['greedy.pkl', 'mdp.pkl']
    x_axis = ['Greedy', 'MDP']

    all_results = []
    for file in filenames:
        with open('./decentralized_exploration/results/two_robots_map_4/'+file, 'rb') as infile:
            all_results.append(plot_one_set(results=pickle.load(infile), plot=False))
            print(all_results[-1].items())
    
    fig = plt.figure()
    ax = fig.add_subplot('111')
    
    local_interactions, = ax.plot(x_axis, [results['local_interactions'] for results in all_results], marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Cumulated iterations with local interactions')
    to_75_pc, = ax.plot(x_axis, [results['to_75_pc'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 75% explored')
    to_90_pc, = ax.plot(x_axis, [results['to_90_pc'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 90% explored')
    total_iterations, = ax.plot(x_axis, [results['total_iterations'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 100% explored')

    plt.legend(handles=[local_interactions, to_75_pc, to_90_pc, total_iterations])
    ax.set_ylim(ymin=0)
    plt.show()
