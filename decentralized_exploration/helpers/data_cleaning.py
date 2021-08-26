import pandas as pd
import cPickle as pickle


def load_data(filename):
    with open('./decentralized_exploration/results/{}.pkl'.format(filename), 'rb') as infile:
        results = pickle.load(infile)
    
    return results


def create_data_entry(data):
    data_dict = {   'local_int': 0, 
                    'time': 0,
                    'total_dist_1': 0,
                    'total_dist_2': 0,
                    'total_dist_3': 0,
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
    data_dict['total_dist_1'] += data[-1][2][0]
    data_dict['total_dist_2'] += data[-1][2][1]
    data_dict['total_dist_3'] += data[-1][2][2]


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
    algorithms = ['greedy', 'utility', 'mdp']
    maps = range(1, 11)
    all_starting_poses = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    idx = pd.MultiIndex.from_product([algorithms, maps, all_starting_poses], names=['Algorithm', 'Map', 'Starting Pose'])
    col = [ 'local_int', 
            'time',
            'total_dist_1',
            'total_dist_2',
            'total_dist_3',
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
                filename = '{}_{}_{}_{}fc_{}iters'.format(algorithm, map_num, starting_pose, communication_level, down_iterations)
                data = load_data(filename)
                data_dict = create_data_entry(data)

                all_data.loc[algorithm, map_num, starting_pose] = data_dict

    all_data.to_pickle('./decentralized_exploration/results/all_data_{}fc_{}iters.pkl'.format(communication_level, down_iterations))
    return all_data

if __name__ == '__main__':
    all_data = create_full_dataframe(communication_level=25, down_iterations=5)
    all_data = create_full_dataframe(communication_level=50, down_iterations=5)
    all_data = create_full_dataframe(communication_level=25, down_iterations=10)
    all_data = create_full_dataframe(communication_level=50, down_iterations=10)
