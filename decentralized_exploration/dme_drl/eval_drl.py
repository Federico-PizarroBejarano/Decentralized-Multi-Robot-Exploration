import numpy as np


np.set_printoptions(linewidth=150)
import torch as th
th.set_printoptions(profile="full", linewidth=150)
import random
random.seed(1234)

from copy import copy, deepcopy
import yaml


from decentralized_exploration.dme_drl.constants import MODEL_DIR, CONFIG_PATH
from decentralized_exploration.dme_drl.world import World
from decentralized_exploration.dme_drl.maddpg.MADDPG import MADDPG

n_agents = 3
n_actions = 8
dim_pose = 2
max_steps = 100
runs = 10

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
print('Evaluate DRL!')
for run in range(runs):
    np.random.seed(run)
    th.manual_seed(run)
    world = World()
    header = ['map_id','steps']
    data = []
    maddpg = MADDPG(n_agents, n_agents, n_actions, dim_pose, 0, 0, -1)
    load_model(maddpg)

    for i_episode in range(20):
        try:
            obs,pose = world.reset(random=False)
            pose = th.tensor(pose)
        except Exception as e:
            continue
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
            obs_, reward, done, _, next_pose, action = world.step(maddpg, obs_history.clone().detach(),                                           pose.clone().detach())

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

