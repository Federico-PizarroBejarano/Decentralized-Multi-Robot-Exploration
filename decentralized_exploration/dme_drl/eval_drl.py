import os

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


from decentralized_exploration.dme_drl.constants import MODEL_DIR, CONFIG_PATH, RESULTS_PATH
from decentralized_exploration.dme_drl.eval_world import EvalWorld
from decentralized_exploration.dme_drl.maddpg.MADDPG import MADDPG

n_agents = 3
n_actions = 8
dim_pose = 2
max_steps = 100
map_ids = 10

def load_model(maddpg):
        checkpoints = th.load(MODEL_DIR + 'model-%d-1.pth' % (config['robots']['number']))
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
                            # 'TL':[(0, 0), (1, 0), (0, 1)],
                            # 'TR':[(0, 19), (1, 19), (0, 18)],
                            'BL':[(19, 0), (18, 0), (19, 1)],
                            # 'BR':[(19, 19), (18, 19), (19, 18)]
                        }
runs = 1



for map_id in range(2, 3):
    for starting_poses_key in all_starting_poses.keys():
        for probability_of_communication_success in [0, 50, 80, 100]:
            run_result_path = RESULTS_PATH + '{}/{}/{}/'.format(probability_of_communication_success, 'dme-drl', '{}-{}'.format(map_id, starting_poses_key))
            os.makedirs(run_result_path, exist_ok=True)
            for probability_of_failed_scan in [10]:
               for failed_communication_interval in [7]:
                    for run in range(runs):
                        try:
                            obs,pose = eval_world.reset('test_{}.npy'.format(map_id), all_starting_poses[starting_poses_key], probability_of_failed_scan, 100 - probability_of_communication_success - 1, failed_communication_interval)
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
                            obs_, reward, done, info, next_pose, action = eval_world.step(maddpg, obs_history.clone().detach(), pose.clone().detach())

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

                        most_steps = 0

                        for robot in eval_world.robots:
                            robot.pose_history = np.array(robot.pose_history)
                            robot.map_history = np.array(robot.map_history)
                            most_steps = max(most_steps, len(robot.map_history))

                        bitmap_history = np.zeros((most_steps, 20, 20)).astype('uint8')
                        map_history = (np.ones((most_steps, 20, 20))*eval_world.config['color']['uncertain'])

                        for robot in eval_world.robots:
                            steps = len(robot.pose_history)
                            robot.pose_history = np.pad(robot.pose_history, [(0,most_steps-steps),(0,0)], 'edge')
                            robot.map_history = np.pad(robot.map_history, [(0, most_steps-steps), (0,0), (0,0)], 'edge')
                            bitmap_history = np.bitwise_or(bitmap_history, robot.map_history!=eval_world.config['color']['uncertain'])
                            if robot.id == 0:
                                pose_history = robot.pose_history[:, np.newaxis, :]
                            else:
                                pose_history = np.concatenate((pose_history, robot.pose_history[:, np.newaxis, :]), axis=1)
                        idx = np.where(bitmap_history == 1)
                        maze_history = np.repeat(eval_world.maze[np.newaxis, :, :], most_steps, axis=0)
                        map_history[idx] = maze_history[idx]
                        idx = np.where(map_history == eval_world.config['color']['uncertain'])
                        map_history[idx] = -1
                        idx = np.where(map_history == eval_world.config['color']['obstacle'])
                        map_history[idx] = 1

                        np.save(run_result_path+'robot_poses', pose_history)
                        np.save(run_result_path+'pixel_maps', map_history)
                        print(np.unique(map_history))

                        exit()
