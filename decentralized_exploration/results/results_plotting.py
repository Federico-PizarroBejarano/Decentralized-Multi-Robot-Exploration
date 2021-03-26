import numpy as np
import matplotlib.pyplot as plt

def plot_all_results(results):
    fig = plt.figure()
    ax = fig.add_subplot('111')

    local_interactions = []
    to_50_pc = []
    to_75_pc = []
    total_iterations = []

    for test in results:
        num_of_local_interactions = np.sum(test[:, 1])
        
        iterations_to_75_pc = 0
        iterations_to_50_pc = 0

        for iteration in range(test.shape[0]):
            if test[iteration][0] > 0.75 and iterations_to_75_pc == 0:
                iterations_to_75_pc = iteration
            elif test[iteration][0] > 0.5 and iterations_to_50_pc == 0:
                iterations_to_50_pc = iteration
        
        local_interactions.append(num_of_local_interactions)
        to_50_pc.append(iterations_to_50_pc)
        to_75_pc.append(iterations_to_75_pc)
        total_iterations.append(test.shape[0])

    local_interactions = ax.plot(range(1, len(results)+1), local_interactions, label='Cumulated iterations with local interactions')
    to_50_pc = ax.plot(range(1, len(results)+1), to_50_pc, label='Iterations until 50% explored')
    to_75_pc = ax.plot(range(1, len(results)+1), to_75_pc, label='Iterations until 75% explored')
    total_iterations = ax.plot(range(1, len(results)+1), total_iterations, label='Iterations until 100% explored')

    plt.legend(handles=[local_interactions, to_50_pc, to_75_pc, total_iterations])
    ax.set_ylim(ymin=0)
    plt.show()
