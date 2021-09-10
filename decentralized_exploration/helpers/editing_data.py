import os
import numpy as np
import pickle

import matplotlib.pyplot as plt

def rename_files():
    for filename in os.listdir('./decentralized_exploration/results'):
        if '_0fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('_0fc', '_100fc')
            os.rename(src, dst)
        elif '10fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('10fc', '90fc')
            os.rename(src, dst)
        elif '20fc' in filename:
            src ='./decentralized_exploration/results/'+ filename
            dst ='./decentralized_exploration/results/'+ filename.replace('20fc', '80fc')
            os.rename(src, dst)
  

def delete_files():
    for filename in os.listdir('./decentralized_exploration/results'):
        if 'rerun' not in filename and 'trajectories' not in filename:
            os.remove('./decentralized_exploration/results/'+filename)
  

def fix_distances():
    for filename in os.listdir('./decentralized_exploration/results'):
        if 'greedy' in filename or 'utility' in filename or 'mdp' in filename:
            with open('./decentralized_exploration/results/{}'.format(filename), 'rb') as infile:
                results = pickle.load(infile)

            if 'top_right' in filename:
                last_positions = [(0, 19), (1, 19), (0, 18)]
            elif 'top_left' in filename:
                last_positions = [(0, 0), (1, 0), (0, 1)]
            elif 'bottom_right' in filename:
                last_positions = [(19, 19), (18, 19), (19, 18)]
            elif 'bottom_left' in filename:
                last_positions = [(19, 0), (18, 0), (19, 1)]

            distances_travelled = [0, 0, 0]

            for it in range(len(results)):
                for robot in [0, 1, 2]:
                    distances_travelled[robot] += np.linalg.norm(np.array(last_positions[robot]) - np.array(results[it][3+robot]))
                    last_positions[robot] = results[it][3+robot]
                
                results[it][2] = list(distances_travelled)
            
            with open('./decentralized_exploration/results/'+filename, 'wb') as outfile:
                pickle.dump(results, outfile, pickle.HIGHEST_PROTOCOL)


def reformat_pickles():
    all_starting_poses = {  
                        'TL':[(0, 0), (1, 0), (0, 1)], 
                        'TR':[(0, 19), (1, 19), (0, 18)], 
                        'BL':[(19, 0), (18, 0), (19, 1)], 
                        'BR':[(19, 19), (18, 19), (19, 18)] 
                    }

    for comm_success in [0, 50, 80, 100]:
        for algo in ['made-net', 'made-net-dt']:
            for trial in os.listdir('./decentralized_exploration/results/trajectories/{}/{}'.format(comm_success, algo)):
                print(comm_success, algo, trial)
                with open('./decentralized_exploration/results/trajectories/{}/{}/{}/agt_traj.pickle'.format(comm_success, algo, trial), 'rb') as infile:
                    poses = pickle.load(infile)
                with open('./decentralized_exploration/results/trajectories/{}/{}/{}/scanned_cells.pickle'.format(comm_success, algo, trial), 'rb') as infile:
                    masks = pickle.load(infile)
                
                world_map = np.load('./decentralized_exploration/maps/test_{}.npy'.format(trial.split('-')[0]))
                # plt.imshow(world_map)
                # plt.show()
                pixel_maps = []

                for mask in masks:
                    pixel_map = -np.ones((20, 20))

                    for cell in mask:
                        pixel_map[cell[0], cell[1]] = world_map[cell[0], cell[1]]
                    
                    pixel_maps.append(pixel_map)
                    # plt.imshow(pixel_map, cmap='gray')
                    # plt.pause(0.05)
                
                all_robot_poses = []
                
                for pose in range(len(poses[0])):
                    robot_poses = []
                    for robot in range(len(poses)):
                        robot_poses.append(poses[robot][pose])
                    all_robot_poses.append(robot_poses)


                all_robot_poses.insert(0, all_starting_poses[trial[-2:]])
                all_robot_poses.insert(0, all_starting_poses[trial[-2:]])
                pixel_maps.insert(0, -np.ones((20, 20)))
                pixel_maps.append(world_map)

                np.save('./decentralized_exploration/results/trajectories/{}/{}/{}/robot_poses.npy'.format(comm_success, algo, trial), all_robot_poses)
                np.save('./decentralized_exploration/results/trajectories/{}/{}/{}/pixel_maps.npy'.format(comm_success, algo, trial), pixel_maps)


if __name__ == '__main__':
    reformat_pickles()
