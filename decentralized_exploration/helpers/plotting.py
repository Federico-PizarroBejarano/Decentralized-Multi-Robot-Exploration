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
    max_value = -float('inf')

    if mode == 'value':
        for hexagon in all_hexes:
            if hexagon.V != 0 and hexagon.state == 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.V, 1)
                if abs(round(hexagon.V, 1)) > max_value:
                    max_value = abs(round(hexagon.V, 1))
    elif mode == 'reward':
        for hexagon in all_hexes:
            if hexagon.reward != 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.reward, 1)
                if abs(round(hexagon.reward, 1)) > max_value:
                    max_value = abs(round(hexagon.reward, 1))
    if mode == 'probability':
        for hexagon in all_hexes:
            if hexagon.probability != 0:
                y = 2. * np.sin(np.radians(60)) * (hexagon.r - hexagon.s) / 3.
                rewards[(hexagon.q, y)] = round(hexagon.probability*100, 2)
                if abs(round(hexagon.probability*100, 2)) > max_value:
                    max_value = abs(round(hexagon.probability*100, 2))

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
            # plot.text(x, y, int(round(rewards[(x, y)])), ha='center', va='center', size=8)
            alpha = abs(rewards[(x, y)]/max_value)
            if rewards[(x, y)] > 0:
                c = 'green'
            elif rewards[(x, y)] < 0:
                c = 'red'

        if (x, y) in hex_robot_states:
            alpha = 1.0
            c = 'yellow'
            # plot.text(x, y, hex_robot_states[(x, y)][1][-1], ha='center', va='center', size=8)

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


def plot_one_set(filename, plot=True):
    with open('./decentralized_exploration/results/{}.pkl'.format(filename), 'rb') as infile:
        results = pickle.load(infile)

    local_interactions = []
    to_75_pc = []
    to_90_pc = []
    to_99_pc = []

    for test in results:
        total_iterations = test.shape[0] 
        iterations_to_99_pc = total_iterations
        iterations_to_90_pc = total_iterations
        iterations_to_75_pc = total_iterations

        for iteration in range(test.shape[0]):
            if test[iteration][0]/0.93 > 0.99 and iterations_to_75_pc == total_iterations:
                iterations_to_99_pc = iteration
            elif test[iteration][0]/0.93 > 0.90 and iterations_to_90_pc == total_iterations:
                iterations_to_90_pc = iteration
            elif test[iteration][0]/0.93 > 0.75 and iterations_to_75_pc == total_iterations:
                iterations_to_75_pc = iteration

        num_of_local_interactions = np.sum(test[:iterations_to_99_pc, 1])
        local_interactions.append(num_of_local_interactions)
        to_75_pc.append(iterations_to_75_pc)
        to_90_pc.append(iterations_to_90_pc)
        to_99_pc.append(iterations_to_99_pc)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot('111')

        local_interactions, = ax.plot(range(1, len(results)+1), local_interactions, marker='o', linestyle='dashed', linewidth=1.1, markersize=12, label='Cumulated iterations with local interactions')
        to_75_pc, = ax.plot(range(1, len(results)+1), to_75_pc, marker='o', linewidth=1.1, markersize=12, label='Iterations until 75% explored')
        to_90_pc, = ax.plot(range(1, len(results)+1), to_90_pc, marker='o', linewidth=1.1, markersize=12, label='Iterations until 90% explored')
        to_99_pc, = ax.plot(range(1, len(results)+1), to_99_pc, marker='o', linewidth=1.1, markersize=12, label='Iterations until 99% explored')

        plt.legend(handles=[local_interactions, to_75_pc, to_90_pc, to_99_pc])
        ax.set_ylim(ymin=0)
        plt.show()
    else:
        cumulated_results = {}
        cumulated_results['local_interactions'] = sum(local_interactions)/len(results)
        cumulated_results['to_75_pc'] = sum(to_75_pc)/len(results)
        cumulated_results['to_90_pc'] = sum(to_90_pc)/len(results)
        cumulated_results['to_99_pc'] = sum(to_99_pc)/len(results)

        return cumulated_results


def plot_all_results():
    filenames = ['greedy.pkl', 'mdp.pkl', 'greedy_blocked.pkl', 'mdp_blocked.pkl', 'mdp_ind_blocked.pkl', 'greedy_no_comm.pkl', 'mdp_no_comm.pkl']
    x_axis = ['Greedy', 'MDP', 'Greedy - Blocked', 'MDP - Blocked', 'MDP Independent - Blocked', 'Greedy - No Comm.', 'MDP - No Comm']

    all_results = []
    for file in filenames:
        all_results.append(plot_one_set(filename=file, plot=False))
        print(all_results[-1].items())
    
    fig = plt.figure()
    ax = fig.add_subplot('111')
    
    ax.plot(x_axis, [results['local_interactions'] for results in all_results], marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Cumulated iterations with local interactions')
    ax.plot(x_axis, [results['to_75_pc'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 75% explored')
    ax.plot(x_axis, [results['to_90_pc'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 90% explored')
    ax.plot(x_axis, [results['to_99_pc'] for results in all_results], marker='o', linewidth=2, markersize=12, label='Iterations until 99% explored')

    plt.legend()
    ax.set_ylim(ymin=0)
    plt.show()


def plot_all_results_bar():
    greedy_filenames = ['greedy.pkl', 'greedy_blocked.pkl', 'greedy_no_comm.pkl']
    mdp_filenames = ['mdp.pkl', 'mdp_blocked.pkl', 'mdp_no_comm.pkl']
    x_axis = ['', 'Full Communication', '', 'Limited Communication', '', 'No Communication']

    greedy_results = []
    for file in greedy_filenames:
        greedy_results.append(plot_one_set(filename=file, plot=False))
        print('{}: {}'.format(file, greedy_results[-1].items()))
    
    mdp_results = []
    for file in mdp_filenames:
        mdp_results.append(plot_one_set(filename=file, plot=False))
        print('{}: {}'.format(file, mdp_results[-1].items()))
    
    mdp_ind_results = plot_one_set(filename='mdp_ind_blocked', plot=False)
    print('mdp_ind_blocked.pkl: {}'.format(mdp_ind_results))
    
    fig = plt.figure()
    ax = fig.add_subplot('111')

    width = 0.2

    greedy_indices = [-0.1, 0.8, 1.9]
    mdp_indices = [0.1, 1, 2.1]
    
    ax.bar(greedy_indices, [results['to_75_pc'] for results in greedy_results], width=width, color='g', edgecolor='k', linewidth=1.1, label='Iterations until 75% explored')
    ax.bar(greedy_indices, [results['to_90_pc'] - results['to_75_pc'] for results in greedy_results], bottom=[results['to_75_pc'] for results in greedy_results], width=width, color='b', edgecolor='k', linewidth=1.1, label='Iterations until 90% explored')
    ax.bar(greedy_indices, [results['to_99_pc'] - results['to_90_pc'] for results in greedy_results], bottom=[results['to_90_pc'] for results in greedy_results], width=width, color='r', edgecolor='k', linewidth=1.1, label='Iterations until 99% explored')
    
    ax.bar(mdp_indices, [results['to_75_pc'] for results in mdp_results], width=width, color='g', edgecolor='k', linewidth=1.1)
    ax.bar(mdp_indices, [results['to_90_pc'] - results['to_75_pc'] for results in mdp_results], bottom=[results['to_75_pc'] for results in mdp_results], width=width, color='b', edgecolor='k', linewidth=1.1)
    ax.bar(mdp_indices, [results['to_99_pc'] - results['to_90_pc'] for results in mdp_results], bottom=[results['to_90_pc'] for results in mdp_results], width=width, color='r', edgecolor='k', linewidth=1.1)
    
    ax.bar(1.2, mdp_ind_results['to_75_pc'], width=width, color='g', edgecolor='k', linewidth=1.1)
    ax.bar(1.2, mdp_ind_results['to_90_pc'] - mdp_ind_results['to_75_pc'], bottom=mdp_ind_results['to_75_pc'], width=width, color='b', edgecolor='k', linewidth=1.1)
    ax.bar(1.2, mdp_ind_results['to_99_pc'] - mdp_ind_results['to_90_pc'], bottom=mdp_ind_results['to_90_pc'], width=width, color='r', edgecolor='k', linewidth=1.1)

    li_str = 'local_interactions'
    li = [greedy_results[0][li_str], mdp_results[0][li_str], greedy_results[1][li_str], mdp_results[1][li_str], mdp_ind_results[li_str], greedy_results[2][li_str], mdp_results[2][li_str]]
    ax.scatter([-0.1, 0.1, 0.8, 1.0, 1.2, 1.9, 2.1], li, marker='o', color='y', label='Cumulated local interactions', zorder=1000)

    ax.axes.set_xticklabels(x_axis)

    plt.legend()
    ax.set_ylim(ymin=0)
    plt.show()


def plot_trajectory(filename, map_file='large_map_4'):
    pixel_map = np.load('./decentralized_exploration/maps/{}.npy'.format(map_file))
    with open('./decentralized_exploration/results/{}.pkl'.format(filename), 'rb') as infile:
        results = pickle.load(infile)
        
    robot_1_traj = np.array([r[2] for r in results])
    robot_2_traj = np.array([r[3] for r in results])

    fig = plt.figure()
    ax = fig.add_subplot('111')

    shaded_map = -1*pixel_map - (pixel_map == -1).astype(int)*1.5
    ax.imshow(shaded_map, cmap='gray')

    ax.plot(robot_1_traj[:, 1], robot_1_traj[:, 0], c='r')
    ax.plot(robot_2_traj[:, 1], robot_2_traj[:, 0], c='b')

    plt.show()
