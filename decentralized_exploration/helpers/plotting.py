import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.lines import Line2D
import cPickle as pickle


def plot_grid(grid, plot, robot_states = {}, mode='value'):
    """
    Plots a given Grid. If a robot_pos is given, will highlight the cell the robot is in in red

    Parameters
    ----------
    grid (Grid): the grid to be plotted
    plot (matplotlib.axes): a matplotlib axes object to be plotted on
    robot_states (dict): an optional dictionary where the keys are the robot_ids and the values are RobotStates
    mode (str) = either 'value' to show the value of each cell, 'reward' to show the reward at each cell, 
        'probability' to show the probability of each robot exploring neighboring states, or blank to show nothing
    """

    plot.cla()

    all_cells = grid.all_cells.values()
    colors_list = ['0.5', '1', '0']
    x_coord = [k[1] for k in grid.all_cells.keys()]
    y_coord = [k[0] for k in grid.all_cells.keys()]
    colors = [colors_list[cell.state+1] for cell in all_cells]

    rewards = {}
    max_value = -float('inf')

    if mode == 'value':
        for cell in all_cells:
            if cell.V != 0 and cell.state == 0:
                rewards[cell.coord] = round(cell.V, 1)
                if abs(round(cell.V, 1)) > max_value:
                    max_value = abs(round(cell.V, 1))
    elif mode == 'reward':
        for cell in all_cells:
            if cell.reward != 0:
                rewards[cell.coord] = round(cell.reward, 1)
                if abs(round(cell.reward, 1)) > max_value:
                    max_value = abs(round(cell.reward, 1))
    if mode == 'probability':
        for cell in all_cells:
            if cell.probability != 0:
                rewards[cell.coord] = round(cell.probability, 1)
                if abs(round(cell.probability*100, 2)) > max_value:
                    max_value = abs(round(cell.probability*100, 2))

    plot.set_aspect('equal')

    pixel_robot_states = {}
    for robot in robot_states.keys():
        point=robot_states[robot].pixel_position
        pixel_robot_states[point] = robot[-1]
    
    # Add some coloured cells
    for x, y, c in zip(x_coord, y_coord, colors):  
        alpha = 0.75      
        if (y, x) in rewards:
            alpha = abs(rewards[(y, x)]/max_value)
            if rewards[(y, x)] > 0:
                c = 'green'
            elif rewards[(y, x)] < 0:
                c = 'red'
        if (y, x) in pixel_robot_states:
            c = 'y'
            alpha = 1
            plot.text(x, y, pixel_robot_states[(y, x)], ha='center', va='center', size=8)
        
        plot.scatter(x, y, color=c, alpha=alpha, marker='s', s=140)

    plot.set_xlim(-0.5, 19.5)
    plot.set_ylim(-0.5, 19.5)

    # legend_elements = [Line2D([0], [0], marker='H', markerfacecolor='1', alpha=0.5, color='k', markersize=15, linewidth=0, label='Free Space'), 
    #                     Line2D([0], [0], marker='H', markerfacecolor='0.5', alpha=0.5, color='k', markersize=15, linewidth=0,label='Unknown Space'),
    #                     Line2D([0], [0], marker='H', markerfacecolor='0', alpha=0.5, color='k', markersize=15, linewidth=0, label='Occupied Space'),
    #                     Line2D([0], [0], marker='H', markerfacecolor='g', alpha=0.7, color='k', markersize=15, linewidth=0, label='Value'),
    #                     Line2D([0], [0], marker='H', markerfacecolor='yellow', alpha=1, color='k', markersize=15, linewidth=0, label='Robot')]

    # plot.legend(handles=legend_elements, framealpha=0.9, loc='lower right')

    plot.invert_yaxis()


def plot_map(pixel_map, plot, robot_pos=[]):
    """
    Plots a pixel map

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


def process_one_test(filename):
    """
    Takes the filename of a test and outputs a dictionary of relevant test results

    Parameters
    ----------
    filename (str): the filename of the test, without the file extension

    Returns
    -------
    cumulated_results (dict): a dictionary containing:
        'local_interactions': average number of local interactions
        'to_75_pc': average number of iterations until 75% explored
        'to_90_pc': average number of iterations until 90% explored
        'to_99_pc': average number of iterations until 99% explored

        'local_interactions_std': standard deviation in 'local_interactions'
        'to_75_pc_std': standard deviation in 'to_75_pc'
        'to_90_pc_std': standard deviation in 'to_90_pc'
        'to_99_pc_std': standard deviation in 'to_99_pc'
    """

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

    cumulated_results = {}
    cumulated_results['local_interactions'] = sum(local_interactions)/len(results)
    cumulated_results['to_75_pc'] = sum(to_75_pc)/len(results)
    cumulated_results['to_90_pc'] = sum(to_90_pc)/len(results)
    cumulated_results['to_99_pc'] = sum(to_99_pc)/len(results)

    cumulated_results['local_interactions_std'] = np.std(np.array(local_interactions))
    cumulated_results['to_75_pc_std'] = np.std(np.array(to_75_pc))
    cumulated_results['to_90_pc_std'] = np.std(np.array(to_90_pc))
    cumulated_results['to_99_pc_std'] = np.std(np.array(to_99_pc))

    return cumulated_results


def plot_local_interactions():
    """
    Plots the average number of local interactions for every test
    """

    greedy_filenames = ['greedy', 'greedy_blocked', 'greedy_no_comm']
    mdp_filenames = ['mdp', 'mdp_blocked', 'mdp_no_comm']
    x_axis = ['', 'Greedy', 'MDP', '', 'Greedy', 'MDP', 'MDP - Ind', '', 'Greedy', 'MDP']

    greedy_results = []
    for file in greedy_filenames:
        greedy_results.append(process_one_test(filename=file))
    
    mdp_results = []
    for file in mdp_filenames:
        mdp_results.append(process_one_test(filename=file))

    mdp_ind_results = process_one_test(filename='mdp_ind_blocked')

    fig = plt.figure()
    ax = fig.add_subplot('111')

    li_str = 'local_interactions'
    li = [greedy_results[0][li_str], mdp_results[0][li_str], greedy_results[1][li_str], mdp_results[1][li_str], mdp_ind_results[li_str], greedy_results[2][li_str], mdp_results[2][li_str]]

    ax.bar([1, 2, 4, 5, 6, 8, 9], li, color='steelblue')

    ax.text(0.6, -10, "Full Communication", weight="bold", fontsize=15)
    ax.text(3.95, -10, "Limited Communication", weight="bold", fontsize=15)
    ax.text(7.55, -10, "No Communication", weight="bold", fontsize=15)

    plt.xticks(range(10), x_axis, fontsize=14)
    ax.set_ylabel('Cumulated Local Interactions\n(# of Iterations)', weight = 'bold', fontsize=17)

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    plt.show()


def plot_computation_time():
    """
    Plots the average computation time for every test
    """

    x_axis = ['', 'Greedy', 'MDP', '', 'Greedy', 'MDP', 'MDP - Ind', '', 'Greedy', 'MDP']

    fig = plt.figure()
    ax = fig.add_subplot('111')

    it_time = np.array([3.464, 8.747, 3.160, 6.241, 6.120, 3.164, 6.074]) - 2.213
    ax.bar([1, 2, 4, 5, 6, 8, 9], it_time, color='plum')

    ax.text(0.6, -0.65, "Full Communication", weight="bold", fontsize=15)
    ax.text(3.95, -0.65, "Limited Communication", weight="bold", fontsize=15)
    ax.text(7.55, -0.65, "No Communication", weight="bold", fontsize=15)

    plt.xticks(range(10), x_axis, fontsize=14)
    ax.set_ylabel('Computation Time per Iteration (s)', weight='bold', fontsize=17, labelpad=10)

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    plt.show()


def plot_variation():
    """
    Plots the average variation in number of local interactions and total mission time for every test
    """

    greedy_filenames = ['greedy', 'greedy_blocked', 'greedy_no_comm']
    mdp_filenames = ['mdp', 'mdp_blocked', 'mdp_no_comm']
    x_axis = ['', 'Greedy', 'MDP', '', 'Greedy', 'MDP', 'MDP - Ind', '', 'Greedy', 'MDP']

    greedy_results = []
    for file in greedy_filenames:
        greedy_results.append(process_one_test(filename=file))
    
    mdp_results = []
    for file in mdp_filenames:
        mdp_results.append(process_one_test(filename=file))

    mdp_ind_results = process_one_test(filename='mdp_ind_blocked')

    fig = plt.figure()
    ax = fig.add_subplot('111')

    std_99_str = 'to_99_pc_std'
    std_99 = [greedy_results[0][std_99_str], mdp_results[0][std_99_str], greedy_results[1][std_99_str], mdp_results[1][std_99_str], mdp_ind_results[std_99_str], greedy_results[2][std_99_str], mdp_results[2][std_99_str]]
    li_str = 'local_interactions_std'
    li = [greedy_results[0][li_str], mdp_results[0][li_str], greedy_results[1][li_str], mdp_results[1][li_str], mdp_ind_results[li_str], greedy_results[2][li_str], mdp_results[2][li_str]]

    ax.bar(np.array([1, 2, 4, 5, 6, 8, 9])+0.2, std_99, width=0.4, color='tomato')
    ax.bar(np.array([1, 2, 4, 5, 6, 8, 9])-0.2, li, width=0.4, color='cornflowerblue')

    ax.text(0.6, -9, "Full Communication", weight="bold", fontsize=15)
    ax.text(3.95, -9, "Limited Communication", weight="bold", fontsize=15)
    ax.text(7.55, -9, "No Communication", weight="bold", fontsize=15)

    plt.xticks(range(10), x_axis, fontsize=14)
    ax.set_ylabel('Number of Iterations', weight = 'bold', fontsize=17)

    legend_elements = [ Line2D([0], [0], color='cornflowerblue', lw=3, label='STD of Local Interations'), 
                        Line2D([0], [0], color='tomato', lw=3, label='STD of Mission Time'),]

    ax.legend(handles=legend_elements, framealpha=0.95, fontsize=13)

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    plt.show()


def plot_exploration_rate():
    """
    Plots the average percent of area explored per number of iterations for every test
    """

    files = ['mdp', 'mdp_blocked', 'mdp_ind_blocked', 'mdp_no_comm', 'greedy', 'greedy_blocked', 'greedy_no_comm']
    labels = ['MDP - Full Communication', 'MDP - Limited Communication', 'MDP Ind - Limited Communication', 'MDP - No Communication', 'Greedy - Full Communication', 'Greedy - Limited Communication', 'Greedy - No Communication']

    results = []
    to_99_pc = []
    for file in files:
        with open('./decentralized_exploration/results/{}.pkl'.format(file), 'rb') as infile:
            results.append(pickle.load(infile))
            to_99_pc.append(process_one_test(filename=file)['to_99_pc'])
    
    fig = plt.figure()
    ax = fig.add_subplot('111')

    for result in range(len(results)):
        percent_explored = [0 for i in range(to_99_pc[result])]
        runs_hit = [0 for i in range(to_99_pc[result])]

        for i in range(10):
            for it in range(to_99_pc[result]):
                if it < results[result][i].shape[0] and results[result][i][it, 0]/0.93 < 0.99:
                    percent_explored[it] += results[result][i][it, 0]/0.93
                    runs_hit[it] += 1
        
        percent_explored = [percent_explored[i]/runs_hit[i]*100 for i in range(len(percent_explored))]
    
        if 'Greedy' in labels[result]:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        
        if 'Ind' in labels[result]:
            color = 'violet'
        elif 'Full' in labels[result]:
            color = 'r'
        elif 'Limited' in labels[result]:
            color = 'blue'
        else:
            color = 'green'

        ax.plot(range(1, len(percent_explored)+1), percent_explored, color=color, linestyle=linestyle, linewidth=2, label=labels[result])

    plt.legend(loc='lower right', fontsize=13)
    ax.grid(True)
    ax.set_xlabel('Number of Iterations', weight = 'bold', fontsize=17)
    ax.set_ylabel('Percent Explored (%)', weight = 'bold', fontsize=17)

    plt.show()


def plot_trajectory(filename, map_file='large_map_4'):
    """
    Plots the trajectories of the two robots in a sample run on the map

    Parameters
    ----------
    filename (str): the filename of the sample test run, without the file extension
    map_file (str): the name of the map to be plotted on
    """

    pixel_map = np.load('./decentralized_exploration/maps/{}.npy'.format(map_file))

    with open('./decentralized_exploration/results/trajectories/{}.pkl'.format(filename), 'rb') as infile:
        results = pickle.load(infile)
    
    interactions = [r[1] for r in results]
    robot_1_traj = [r[2] for r in results]
    robot_2_traj = [r[3] for r in results]

    robot_1_interactions = []
    robot_2_interactions = []

    fig = plt.figure()
    ax = fig.add_subplot('111')

    starting_area = Circle([300, 500], radius=75, color='g', alpha=0.2, label='Starting area')
    ax.add_patch(starting_area)

    for i in range(len(interactions)):
        if interactions[i] == True:
            robot_1_interactions.append(robot_1_traj[i])
            robot_2_interactions.append(robot_2_traj[i])
        else:
            if len(robot_1_interactions) != 0:
                robot_1_interactions = np.array(robot_1_interactions)
                robot_2_interactions = np.array(robot_2_interactions)
                ax.plot(robot_1_interactions[:, 1], robot_1_interactions[:, 0], c='yellow', linewidth=8)
                ax.plot(robot_2_interactions[:, 1]+2.5, robot_2_interactions[:, 0]+2.5, c='yellow', linewidth=8)
                robot_1_interactions = []
                robot_2_interactions = []
            

    robot_1_traj = np.array(robot_1_traj)
    robot_2_traj = np.array(robot_2_traj)

    if interactions[0] == True:
        ax.plot(robot_1_traj[0, 1], robot_1_traj[0, 0], marker='o', markersize=15, color='yellow')
        ax.plot(robot_2_traj[0, 1]+2.5, robot_2_traj[0, 0]+2.5, marker='o', markersize=15, color='yellow')

    shaded_map = -1*pixel_map - (pixel_map == -1).astype(int)*1.5
    ax.imshow(shaded_map, cmap='gray')

    NPOINTS = robot_1_traj[:, 1].shape[0]

    cm_red = plt.get_cmap('Reds')
    ax.set_color_cycle([cm_red(1.*i/(NPOINTS*2)) for i in range(NPOINTS, NPOINTS*2)])
    for i in range(NPOINTS-1):
        plt.plot(list(robot_1_traj[:, 1])[i:i+2], list(robot_1_traj[:, 0])[i:i+2], linewidth=3)

    start_loc, = ax.plot(robot_1_traj[0, 1], robot_1_traj[0, 0], marker='o', markersize=10, color=cm_red(1/2.0), linewidth=0, label='Start location')
    end_loc, = ax.plot(robot_1_traj[-1, 1], robot_1_traj[-1, 0], marker='*', markersize=13, color=cm_red(1.0), linewidth=0, label='End location')

    cm_blue = plt.get_cmap('Blues')
    ax.set_color_cycle([cm_blue(1.*i/(NPOINTS*2)) for i in range(NPOINTS, NPOINTS*2)])
    for i in range(NPOINTS-1):
        plt.plot(list(robot_2_traj[:, 1]+2.5)[i:i+2], list(robot_2_traj[:, 0]+2.5)[i:i+2], linewidth=3, linestyle='dashed')

    ax.plot(robot_2_traj[0, 1]+2.5, robot_2_traj[0, 0]+2.5, marker='o', markersize=10, color=cm_blue(1/2.0))
    ax.plot(robot_2_traj[-1, 1]+2.5, robot_2_traj[-1, 0]+2.5, marker='*', markersize=13, color=cm_blue(1.0))

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    legend_elements = [ Line2D([0], [0], color=cm_red(1/2.0), lw=3, label='Robot 1'),
                        Line2D([0], [0], color=cm_blue(1/2.0), lw=3, linestyle='dashed', label='Robot 2'),
                        Line2D([0], [0], color='yellow', lw=8, label='Local interactions'),
                        start_loc, 
                        end_loc,
                        starting_area]

    ax.legend(handles=legend_elements, framealpha=0.95, bbox_to_anchor=(1, 1), loc='upper left', fontsize=13)
    ax.set_ylim(pixel_map.shape[0])

    plt.show()


if __name__ == '__main__':
    plot_computation_time()
    plot_variation()
    plot_local_interactions()

    plot_trajectory('greedy')
    plot_trajectory('greedy_blocked')
    plot_trajectory('greedy_no_comm')

    plot_trajectory('mdp')
    plot_trajectory('mdp_blocked')
    plot_trajectory('mdp_no_comm')

    plot_exploration_rate()
