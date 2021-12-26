import numpy as np
import torch as th
from tensorboardX import SummaryWriter
from copy import copy,deepcopy
from torch.distributions import categorical
import time
import os
import yaml

from decentralized_exploration.dme_drl.constants import render_world, PROJECT_PATH, CONFIG_PATH, MODEL_DIR
from decentralized_exploration.dme_drl.world import World
from decentralized_exploration.dme_drl.maddpg.MADDPG import MADDPG
from decentralized_exploration.dme_drl.sim_utils import onehot_from_action

# tensorboard writer
time_now = time.strftime("%m%d_%H%M%S")
writer = SummaryWriter(PROJECT_PATH + '/runs/' + time_now)

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = World()
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)
n_agents = world.number
n_states = world.number
n_actions = 8
n_pose = 2
# capacity = 1000000
capacity = 5000
# batch_size = 1000
batch_size = 100

n_episode = 200000
# max_steps = 1000
max_steps = 50
# episodes_before_train = 1000
episodes_before_train = 100

win = None
param = None
avg = None
load_model = False

maddpg = MADDPG(n_agents, n_states, n_actions, n_pose, batch_size, capacity,
                episodes_before_train)
with open(CONFIG_PATH, 'r') as stream:
    config = yaml.safe_load(stream)

if load_model:
    if not os.path.exists(MODEL_DIR):
        pass
    else:
        checkpoints = th.load(MODEL_DIR + '/model/model-%d.pth' % (config['robots']['number']))
        for i, actor in enumerate(maddpg.actors):
            actor.load_state_dict(checkpoints['actor_%d' % (i)])
            maddpg.actors_target[i] = deepcopy(actor)
        for i, critic in enumerate(maddpg.critics):
            critic.load_state_dict(checkpoints['critic_%d' % (i)])
            maddpg.critics_target[i] = deepcopy(critic)
        for i, actor_optim in enumerate(maddpg.actor_optimizer):
            actor_optim.load_state_dict(checkpoints['actor_optim_%d' % (i)])
        for i, critic_optim in enumerate(maddpg.critic_optimizer):
            critic_optim.load_state_dict(checkpoints['critic_optim_%d' % (i)])


FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    try:
        obs,pose = world.reset()
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
    obs_history=np.zeros((n_agents,obs.shape[1]*6,obs.shape[2]))
    for i in range(n_agents):
        obs_history[i] = np.vstack((obs_t_minus_0[i],obs_t_minus_1[i],obs_t_minus_2[i],
                            obs_t_minus_3[i],obs_t_minus_4[i],obs_t_minus_5[i]))

    if isinstance(obs, np.ndarray):
        obs_history = th.from_numpy(obs_history).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    wrong_step = 0
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        if render_world:
            world.render()
        obs_history = obs_history.type(FloatTensor)
        action_probs = maddpg.select_action(obs_history, pose).data.cpu()
        action_probs_valid = np.copy(action_probs)
        action = []
        for i,probs in enumerate(action_probs):
            rbt = world.robots[i]
            for j,frt in enumerate(rbt.get_and_update_frontier_by_direction()):
                if len(frt) == 0:
                    action_probs_valid[i][j] = 0
            # if np.array_equal(action_probs_valid[i], np.zeros_like(action_probs_valid[i])):
            #     print(rbt.id, rbt.pose)
            #     print(rbt.slam_map)
            #     np.save('empty_frontier', rbt.slam_map)
            action.append(categorical.Categorical(probs=th.tensor(action_probs_valid[i])).sample())

        action = th.tensor(onehot_from_action(action))
        acts = np.argmax(action,axis=1)
        for i in range(len(acts)):
            if len(world.robots[i].frontier_by_direction[acts[i]]) == 0:
                # NOOP 指令
                acts[i] = -1

        obs_, reward, done, _, next_pose = world.step(acts)
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

        if t != max_steps - 1:
            next_obs_history = th.tensor(obs_history_)
        elif done:
            next_obs_history = None
        else:
            next_obs_history = None
        total_reward += reward.sum()
        rr += reward.cpu().numpy()

        maddpg.memory.push(obs_history, action, next_obs_history, reward, pose, next_pose)
        obs_history = next_obs_history
        pose = next_pose
        if t % 10 == 0:
            c_loss, a_loss = maddpg.update_policy()
        if done:
            break
    exit()

    # if not discard:
    maddpg.episode_done += 1
    if maddpg.episode_done % 100 == 0:
        print('Save Models......')
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        dicts = {}
        for i in range(maddpg.n_agents):
            dicts['actor_%d' % (i)] = maddpg.actors_target[i].state_dict()
            dicts['critic_%d' % (i)] = maddpg.critics_target[i].state_dict()
            dicts['actor_optim_%d' % (i)] = maddpg.actor_optimizer[i].state_dict()
            dicts['critic_optim_%d' % (i)] = maddpg.critic_optimizer[i].state_dict()
        th.save(dicts, MODEL_DIR + '/model-%d.pth' % (config['robots']['number']))
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    # visual
    writer.add_scalars('scalar/reward',{'total_rwd':total_reward,'r0_rwd':rr[0],'r1_rwd':rr[1]},i_episode)
    if i_episode > episodes_before_train and i_episode % 10 == 0:
        writer.add_scalars('scalar/mean_rwd',{'mean_reward':np.mean(reward_record[-100:])}, i_episode)
    if not c_loss is None:
        writer.add_scalars('loss/c_loss',{'r0':c_loss[0],'r1':c_loss[1]},i_episode)
    if not a_loss is None:
        writer.add_scalars('loss/a_loss',{'r0':a_loss[0],'r1':a_loss[1]},i_episode)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

world.close()