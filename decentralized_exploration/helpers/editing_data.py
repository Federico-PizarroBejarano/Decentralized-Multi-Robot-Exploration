import os
import numpy as np
import cPickle as pickle

def rename_files():
    for _, filename in enumerate(os.listdir('./decentralized_exploration/results')):
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
    for _, filename in enumerate(os.listdir('./decentralized_exploration/results')):
        if 'rerun' not in filename:
            os.remove('./decentralized_exploration/results'+filename)
  

def fix_distances():
    for _, filename in enumerate(os.listdir('./decentralized_exploration/results')):
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

if __name__ == '__main__':
    fix_distances()
