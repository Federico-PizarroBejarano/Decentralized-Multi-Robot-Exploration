import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt


def load_data(filename):
    with open('./decentralized_exploration/results/{}.pkl'.format(filename), 'rb') as infile:
        results = pickle.load(infile)
    
    return results


def create_data_entry(data):
    data_dict = {   'local_int': 0, 
                    'time': 0,
                    'total_dist': 0,
                    'to_20': 0, 
                    'to_30': 0, 
                    'to_40': 0, 
                    'to_50': 0, 
                    'to_60': 0, 
                    'to_70': 0, 
                    'to_80': 0, 
                    'to_90': 0, 
                    'to_100': 0 }
    
    data_dict['time'] += data[-1][-1]
    data_dict['total_dist'] += data[-1][2][0] + data[-1][2][1] + data[-1][2][2]

    for iteration in range(len(data)): 
        data_dict['local_int'] += int(len(data[iteration][1])/2)

        if data[iteration][0] >= 1.0 and data_dict['to_100'] == 0:
            data_dict['to_100'] = iteration
        elif data[iteration][0] >= 0.9 and data_dict['to_90'] == 0:
            data_dict['to_90'] = iteration
        elif data[iteration][0] >= 0.8 and data_dict['to_80'] == 0:
            data_dict['to_80'] = iteration
        elif data[iteration][0] >= 0.7 and data_dict['to_70'] == 0:
            data_dict['to_70'] = iteration
        elif data[iteration][0] >= 0.6 and data_dict['to_60'] == 0:
            data_dict['to_60'] = iteration
        elif data[iteration][0] >= 0.5 and data_dict['to_50'] == 0:
            data_dict['to_50'] = iteration
        elif data[iteration][0] >= 0.4 and data_dict['to_40'] == 0:
            data_dict['to_40'] = iteration
        elif data[iteration][0] >= 0.3 and data_dict['to_30'] == 0:
            data_dict['to_30'] = iteration
        elif data[iteration][0] >= 0.2 and data_dict['to_20'] == 0:
            data_dict['to_20'] = iteration
    
    return data_dict


def create_full_dataframe(communication_level, down_iterations):
    algorithms = [
        'greedy', 
        'utility', 
        'mdp'
    ]
    maps = range(1, 11)
    all_starting_poses = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    idx = pd.MultiIndex.from_product([algorithms, maps, all_starting_poses], names=['Algorithm', 'Map', 'Starting Pose'])
    col = [ 'local_int', 
            'time',
            'total_dist',
            'to_20', 
            'to_30', 
            'to_40', 
            'to_50', 
            'to_60', 
            'to_70', 
            'to_80', 
            'to_90', 
            'to_100' ]
    
    all_data = pd.DataFrame('-', index=idx, columns=col) 

    for algorithm in algorithms:
        for map_num in maps:
            for starting_pose in all_starting_poses:
                filename = '{}_{}_{}_{}fc_{}iters_rerun'.format(algorithm, map_num, starting_pose, communication_level, down_iterations)
                data = load_data(filename)
                data_dict = create_data_entry(data)

                all_data.loc[algorithm, map_num, starting_pose] = data_dict

    all_data.to_pickle('./decentralized_exploration/results/all_data_{}fc_{}iters_rerun.pkl'.format(communication_level, down_iterations))
    return all_data


def compare_parameters(communication_levels, down_iterations):
    algorithms = [
        'greedy', 
        'utility', 
        'mdp'
    ]

    idx = pd.MultiIndex.from_product([algorithms, communication_levels], names=['Algorithm', 'Prob FC'])
    col = [ 'local_int', 
            'time',
            'total_dist',
            'to_20', 
            'to_30', 
            'to_40', 
            'to_50', 
            'to_60', 
            'to_70', 
            'to_80', 
            'to_90', 
            'to_100' ]
    
    all_data = pd.DataFrame('-', index=idx, columns=col) 

    for pfc in communication_levels:
        for fci in down_iterations:
            filename = 'all_data_{}fc_{}iters_rerun.pkl'.format(pfc, fci)
            df = pd.read_pickle('./decentralized_exploration/results/'+filename)

            for algorithm in algorithms:
                all_data.loc[algorithm, pfc] = df.loc[algorithm].mean()

    all_data.to_pickle('./decentralized_exploration/results/all_data_summary_rerun.pkl')
    return all_data


def get_exploration_rates(communication_levels, down_iterations):
    algorithms = [
        'greedy', 
        'utility', 
        'mdp'
    ]
    maps = range(1, 11)
    all_starting_poses = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    resolution = 0.0001
    cov_pct = np.arange(0.1, 1.0+resolution, resolution).tolist()

    all_lists = {}
    all_metrics = {}

    for algorithm in algorithms:
        for pfc in communication_levels:
            for fci in down_iterations:
                if pfc == 0:
                    fci = 0
                elif pfc == 100:
                    fci = 5
                
                dist_trav_av = [0]*len(cov_pct)
                for map_num in maps:
                    for starting_pose in all_starting_poses:
                        filename = '{}_{}_{}_{}fc_{}iters'.format(algorithm, map_num, starting_pose, pfc, fci)
                        with open('./decentralized_exploration/results/{}.pkl'.format(filename), 'rb') as infile:
                            results = pickle.load(infile)

                        percent_explored = []
                        distance_travelled = []
                        for it in range(len(results)):
                            percent_explored.append(results[it][0])
                            distance_travelled.append(results[it][2][0] + results[it][2][1] + results[it][2][2])
                        dist_trav_interp = np.interp(cov_pct, percent_explored, distance_travelled)
                        dist_trav_av = [sum(x) for x in zip(dist_trav_interp, dist_trav_av)]
                
                dist_trav_av = [dist/40.0 for dist in dist_trav_av]
                all_lists['{}_{}fc'.format(algorithm, pfc)] = dist_trav_av
                all_metrics['{}_{}fc'.format(algorithm, pfc)] = calculate_objective_function(dist_trav_av)

    with open('./decentralized_exploration/results/all_exploration_rates', 'wb') as outfile:
        pickle.dump([all_lists, all_metrics], outfile, pickle.HIGHEST_PROTOCOL)


def calculate_objective_function(dist_trav):
    resolution = 0.0001
    cov_pct = np.arange(0.1, 1.0+resolution, resolution).tolist()

    obj_metric = 0

    for i in range(len(cov_pct)):
        obj_metric += cov_pct[i]/dist_trav[i]

    return obj_metric


if __name__ == '__main__':  
    # for fci in [7]:
    #     for pfc in [100]:#, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #         create_full_dataframe(communication_level=pfc, down_iterations=fci)
    
    df = compare_parameters([0, 20, 50, 100], [7])

    # df.to_csv('./decentralized_exploration/results/all_data_summary.csv') 

    # get_exploration_rates([0, 20, 50, 100], [7]) 
