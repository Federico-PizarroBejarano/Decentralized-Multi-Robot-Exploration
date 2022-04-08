import os
import pandas as pd
import pickle

import numpy as np
np.random.seed(1234)
np.set_printoptions(linewidth=150)
import torch as th
th.manual_seed(1234)
th.set_printoptions(profile="full", linewidth=150)
import random
random.seed(1234)

from copy import copy, deepcopy
import yaml
from time import time


from decentralized_exploration.dme_drl.constants import MODEL_DIR, CONFIG_PATH, RESULTS_PATH
from decentralized_exploration.dme_drl.eval_world import EvalWorld
from decentralized_exploration.dme_drl.maddpg.MADDPG import MADDPG

n_agents = 3
n_actions = 8
dim_pose = 2
max_steps = 100
map_ids = 10

dir_name = '0323_121251/'

def load_model(maddpg):
        checkpoints = th.load(MODEL_DIR + dir_name + 'model-%d-1.pth' % (config['robots']['number']))
        for i, actor in enumerate(maddpg.actors):
            actor.load_state_dict(checkpoints['actor_%d' % (i)])
            maddpg.actors_target[i] = deepcopy(actor)
        for i, critic in enumerate(maddpg.critics):
            critic.load_state_dict(checkpoints['critic_%d' % (i)])
            maddpg.critics_target[i] = deepcopy(critic)
with open(CONFIG_PATH,'r') as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
    n_agents = config['robots']['number']


FloatTensor = th.FloatTensor
eval_world = EvalWorld()
maddpg = MADDPG(n_agents, n_agents, n_actions, dim_pose, 0, 0, -1)
load_model(maddpg)

all_starting_poses = {
                            'TL':[(0, 0), (1, 0), (0, 1)],
                            'TR':[(0, 19), (1, 19), (0, 18)],
                            'BL':[(19, 0), (18, 0), (19, 1)],
                            'BR':[(19, 19), (18, 19), (19, 18)]
                        }

results = {'map_id':[],
           'starting_pose': [],
           'probability_of_communication_success': [],
           'total_steps': [],
           'r1_steps': [],
           'r2_steps': [],
           'r3_steps': [],
           'distance_travelled':[],
           'r1_distance': [],
           'r2_distance': [],
           'r3_distance': [],
           'local_interactions_parallel': [],
           'local_interactions_sequential': [],
           'local_interactions_avg': [],
           'objective_function': [],
           'times': []}


for probability_of_communication_success in [0, 50, 80, 100]:
    trial = 0
    for starting_poses_key in all_starting_poses.keys():
        for map_id in range(1,11):
            trial += 1
            run_result_path = RESULTS_PATH + '{}/{}/{}/'.format(probability_of_communication_success, 'dme-drl', '{}-{}'.format(map_id, starting_poses_key))
            os.makedirs(run_result_path, exist_ok=True)
            for probability_of_failed_scan in [10]:
                for failed_communication_interval in [7]:
                    total_cumulative_distance = -1
                    while total_cumulative_distance < 100:
                        start = time()
                        try:
                            obs,pose = eval_world.reset('test_{}.npy'.format(map_id), all_starting_poses[starting_poses_key], probability_of_failed_scan - 1, 100 - probability_of_communication_success - 1, failed_communication_interval)
                            pose = th.tensor(pose)
                        except Exception as e:
                            print(e)
                        obs = np.stack(obs)
                        # history initialization
                        obs_t_minus_0 = copy(obs)
                        obs_t_minus_1 = copy(obs)
                        obs_t_minus_2 = copy(obs)
                        obs_t_minus_3 = copy(obs)
                        obs_t_minus_4 = copy(obs)
                        obs_t_minus_5 = copy(obs)
                        obs = th.from_numpy(obs)
                        obs_history = np.zeros((n_agents, obs.shape[1] * 6, obs.shape[2]))
                        for i in range(n_agents):
                            obs_history[i] = np.vstack((obs_t_minus_0[i], obs_t_minus_1[i], obs_t_minus_2[i],
                                                        obs_t_minus_3[i], obs_t_minus_4[i], obs_t_minus_5[i]))
                        if isinstance(obs_history, np.ndarray):
                            obs_history = th.from_numpy(obs_history).float()

                        for t in range(max_steps):
                            obs_history = obs_history.type(FloatTensor)
                            obs_, reward, done, info, next_pose, _ = eval_world.step(maddpg, obs_history.clone().detach(), pose.clone().detach())

                            if info == 'done midstep':
                                break

                            next_pose = th.tensor(next_pose)
                            reward = th.FloatTensor(reward).type(FloatTensor)
                            obs_ = np.stack(obs_)
                            obs_ = th.from_numpy(obs_).float()

                            obs_t_minus_5 = copy(obs_t_minus_4)
                            obs_t_minus_4 = copy(obs_t_minus_3)
                            obs_t_minus_3 = copy(obs_t_minus_2)
                            obs_t_minus_2 = copy(obs_t_minus_1)
                            obs_t_minus_1 = copy(obs_t_minus_0)
                            obs_t_minus_0 = copy(obs_)
                            obs_history_ = np.zeros((n_agents, obs.shape[1] * 6, obs.shape[2]))
                            for i in range(n_agents):
                                obs_history_[i] = np.vstack((obs_t_minus_0[i], obs_t_minus_1[i], obs_t_minus_2[i],
                                                             obs_t_minus_3[i], obs_t_minus_4[i], obs_t_minus_5[i]))
                            if not t == max_steps - 1:
                                next_obs_history = th.tensor(obs_history_)
                            elif done:
                                next_obs_history = None
                            else:
                                next_obs_history = None
                            obs_history=next_obs_history
                            pose = next_pose

                        end = time()

                        most_steps = 0

                        for robot in eval_world.robots:
                            robot.pose_history = np.array(robot.pose_history)
                            robot.area_explored_history = np.array(robot.area_explored_history)
                            most_steps = max(most_steps, robot.sub_time_step)


                        total_explored_area_per_step = np.zeros(most_steps)
                        joint_distance_travelled_per_step = np.zeros_like(total_explored_area_per_step)

                        for robot in eval_world.robots:
                            steps = robot.sub_time_step
                            # objective function
                            robot.pose_history = np.pad(robot.pose_history, [(0, most_steps - steps), (0, 0)], 'edge')
                            robot.area_explored_history = np.pad(robot.area_explored_history, [(0,most_steps-steps)])
                            robot.distance_travelled_history = np.pad(robot.distance_travelled_history, [(0,most_steps-steps)])
                            total_explored_area_per_step += robot.area_explored_history
                            joint_distance_travelled_per_step += robot.distance_travelled_history

                        # objective function
                        cumulative_explored_percentage = np.cumsum(total_explored_area_per_step) / 400

                        # find total interactions
                        total_interactions = 0
                        for step in range(most_steps):
                            interactions = 0
                            for robot1 in eval_world.robots:
                                for robot2 in eval_world.robots:
                                    if robot1.id < robot2.id:
                                        if eval_world.is_local(robot1.pose_history[step], robot2.pose_history[step]):
                                            interactions += 1
                                            break
                                    if interactions == 1:
                                        break
                            total_interactions += interactions

                        # find objective function
                        cumulative_joint_distance = np.cumsum(joint_distance_travelled_per_step)
                        cumulative_explored_percentage_range = np.linspace(0.1,1,num=9001)

                        cumulative_joint_distance_interp = np.interp(x=cumulative_explored_percentage_range, xp=cumulative_explored_percentage, fp=cumulative_joint_distance)

                        objective_function_value = np.sum(cumulative_explored_percentage_range / cumulative_joint_distance_interp)

                        total_cumulative_distance = cumulative_joint_distance[-1]

                        if total_cumulative_distance >= 100:
                            results['map_id'].append(map_id)
                            results['starting_pose'].append(starting_poses_key)
                            results['probability_of_communication_success'].append(probability_of_communication_success)
                            results['total_steps'].append(sum([robot.sub_time_step for robot in eval_world.robots]) / len(eval_world.robots))

                            results['r1_steps'].append(eval_world.robots[0].sub_time_step)
                            results['r2_steps'].append(eval_world.robots[1].sub_time_step)
                            results['r3_steps'].append(eval_world.robots[2].sub_time_step)

                            results['distance_travelled'].append(sum([robot.distance for robot in eval_world.robots]))
                            results['r1_distance'].append(eval_world.robots[0].distance)
                            results['r2_distance'].append(eval_world.robots[1].distance)
                            results['r3_distance'].append(eval_world.robots[2].distance)

                            results['local_interactions_parallel'].append(total_interactions)
                            results['local_interactions_sequential'].append(eval_world.local_interactions)
                            results['local_interactions_avg'].append((total_interactions + eval_world.local_interactions) / 2)
                            results['objective_function'].append(objective_function_value)
                            results['times'].append(end-start)

                            plot_path = RESULTS_PATH + '{}/{}/'.format(probability_of_communication_success, 'dme-drl')
                            distance_by_explored = []

                            assert(len(cumulative_explored_percentage) == len(cumulative_joint_distance))

                            for i in range(len(cumulative_explored_percentage)):
                                distance_by_explored.append((cumulative_joint_distance[i], cumulative_explored_percentage[i]))

                            with open(plot_path + 'trial_{}.pickle'.format(trial), 'wb') as f:
                                pickle.dump(distance_by_explored, f)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

results = pd.DataFrame(results)
averages = results.groupby(by='probability_of_communication_success').mean().reset_index()

trunc_results = results[results['distance_travelled'] > 100]
trunc_averages = trunc_results.groupby(by='probability_of_communication_success').mean().reset_index()


results.to_csv(RESULTS_PATH+'results.csv')
averages.to_csv(RESULTS_PATH+'averages.csv')

trunc_results.to_csv(RESULTS_PATH+'trunc_results.csv')
trunc_averages.to_csv(RESULTS_PATH+'trunc_averages.csv')

