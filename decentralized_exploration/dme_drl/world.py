import os
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch as th

from decentralized_exploration.dme_drl.frontier_utils import merge_frontiers_and_remove_pose,\
    remove_pose_from_frontier, cleanup_frontier
from decentralized_exploration.dme_drl.robot import Robot
from decentralized_exploration.core.robots.utils.field_of_view import bresenham
from decentralized_exploration.dme_drl.constants import render_world, PROJECT_PATH, CONFIG_PATH, manual_check, \
    RESET_WORLD_PATH, ID_TO_COLOR, STEP_WORLD_PATH, STEP_ROBOT_PATH


class World(gym.Env):
    def __init__(self, number=None):
        with open(CONFIG_PATH) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.map_id_set_train = np.loadtxt(PROJECT_PATH + self.config['map_id_train_set'], str)
        self.map_id_set_eval = np.loadtxt(PROJECT_PATH + self.config['map_id_train_set'], str) # change to eval set
        if number is None:
            self.number = self.config['robots']['number']
        else:
            self.number = number
        # parameters will be set in reset func
        self.map_id = None
        self.target_points = []
        self.maze = np.zeros([1, 1])
        self.slam_map = np.zeros_like(self.maze)
        self.after_comm_map = np.zeros_like(self.maze)
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.zeros_like(self.slam_map)
        self.data_transmitted = 0
        self.robots = []
        self.robot_sensor_range = self.config['robots']['sensorRange']
        self.probability_of_failed_communication = self.config['robots']['probabilityOfFailedCommunication']
        self.frontier = set()
        self.episode = -1
        self.time_step = -1
        if render_world or manual_check:
            self.fig = plt.figure('global')
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)

    def record_poses(self, robot1, robot2):
        robot1.poses[:, 2 * robot2.id] = robot2.pose[0]
        robot1.poses[:, 2 * robot2.id + 1] = robot2.pose[1]

        robot2.poses[:, 2 * robot1.id] = robot1.pose[0]
        robot2.poses[:, 2 * robot1.id + 1] = robot1.pose[1]

        robot1.seen_robots.add(robot2.id)
        robot2.seen_robots.add(robot1.id)

        self.local_interactions += 1

    def communicate(self, robot1, robot2):
        if self._can_communicate():
            robot1.render(robot1.render_path + 'r{}_s{}_pre_merge_with_r{}_former'.format(robot1.id, robot1.counter, robot2.id))
            robot2.render(robot1.render_path + 'r{}_s{}_pre_merge_with_r{}_latter'.format(robot1.id, robot1.counter, robot2.id))

            self._merge_maps(robot1, robot2)
            self._merge_frontiers_after_communicate(robot1, robot2)

            robot1.render(robot1.render_path + 'r{}_s{}_pro_merge_with_r{}_former'.format(robot1.id, robot1.counter, robot2.id))
            robot2.render(robot1.render_path + 'r{}_s{}_pro_merge_with_r{}_latter'.format(robot1.id, robot1.counter, robot2.id))


    def reset(self,random=True):
        self.episode += 1
        self.time_step = -1
        self.local_interactions = 0
        if random:
            self.map_id = PROJECT_PATH + np.random.choice(self.map_id_set_train)
        else:
            self.map_id = PROJECT_PATH + self.map_id_set_eval[0]
        print('map id： ',self.map_id)
        self.target_points = []
        self.maze = np.load(self.map_id).astype('uint8')
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.copy(self.maze)
        self.data_transmitted = 0
        self.robots = [Robot(i, np.copy(self.maze)) for i in range(self.number)]

        for rbt in self.robots:
            rbt.robots=self.robots
            rbt.world = self
            rbt.reset(np.copy(self.maze))

        self.slam_map = self._merge_map(self.slam_map)
        self.render(RESET_WORLD_PATH + 'e{}_t{}_pro_reset'.format(self.episode, self.time_step))
        obs_n = []
        pose_n = []

        for id1, robot1 in enumerate(self.robots):
            for id2, robot2 in enumerate(self.robots):
                if id1 < id2:
                    if self.in_range(robot1, robot2):
                        self.record_poses(robot1, robot2)
                        self.local_interactions -= 1 # don't double count local interactions in step 0
            robot1.seen_robots.clear()
            obs_n.append(robot1.get_obs())
            pose_n.append(robot1.get_poses())

        return obs_n,pose_n

    def seed(self, seed=None):
        pass

    def render(self, fname=None):
        if manual_check or render_world:
            global_frontier = set()

            for robot in self.robots:
                global_frontier |= robot.frontier

            for robot in self.robots:
                global_frontier = remove_pose_from_frontier(global_frontier, robot.pose)

            global_frontier = cleanup_frontier(self.slam_map, global_frontier, self.config)

            self.ax.cla()
            self.ax.set_aspect('equal')

            # plot the terrain
            for y in range(self.slam_map.shape[0]):
                for x in range(self.slam_map.shape[1]):
                    val = self.slam_map[y,x]
                    if val == self.config['color']['uncertain']:
                        c = 'gray'
                    if val == self.config['color']['obstacle']:
                        c = 'black'
                    if val == self.config['color']['free']:
                        c = 'white'

                    self.ax.scatter(x, y, color=c, alpha=0.75, marker='s', s=140)

            # plot the robots
            for robot in self.robots:
                self.ax.scatter(robot.pose[1], robot.pose[0], color=ID_TO_COLOR[robot.id], marker='s', alpha=1, s=140)
                self.ax.text(robot.pose[1], robot.pose[0], s=robot.id, ha='center', va='center', size=8)

            for node in global_frontier:
                self.ax.text(node[1], node[0], 'F', ha='center', va='center', size=8)

            self.ax.set_xlim(-0.5, 19.5)
            self.ax.set_ylim(-0.5, 19.5)
            if manual_check:
                self.fig.savefig(fname)
                # np.save(fname, self.slam_map)
            elif render_world:
                plt.pause(0.5)

    def step(self, maddpg, obs_history, pose):
        self.time_step += 1
        obs_n = []
        rwd_n = []
        info_n = []
        pose_n = []
        action_n = None

        self.render(STEP_WORLD_PATH + 'e{}_t{}_pre_step'.format(self.episode, self.time_step))

        for robot in self.robots:

            self.slam_map = self._merge_map(self.slam_map)
            done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95

            if done:
                return None, None, None, 'done midstep', None, None

            rwd, info = robot.step(maddpg, obs_history, pose)

            if info == 'empty frontier':
                return None, None, None, info, None, None

            rwd_n.append(rwd)
            info_n.append(info)
            obs_n.append(robot.get_obs())
            if robot.id == 0:
                action_n = robot.get_action()
            else:
                action_n = th.cat((action_n,robot.get_action()))

        # get accurate poses
        for id1, robot1 in enumerate(self.robots):
            for id2, robot2 in enumerate(self.robots):
                if id1 < id2:
                    if self.in_range(robot1, robot2):
                        self.record_poses(robot1, robot2)
                        self.local_interactions -= 1 # don't double count local interactions in step 0
            robot1.seen_robots.clear()
            pose_n.append(robot1.get_poses())


        maddpg.steps_done += 1

        self.render(STEP_WORLD_PATH + 'e{}_t{}_pro_step'.format(self.episode, self.time_step))

        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95
        return obs_n,rwd_n,done,info_n,pose_n, action_n

    def _can_communicate(self):
        return np.random.randint(100) > self.probability_of_failed_communication

    def in_range(self, robot1, robot2):
        distance = max(abs(robot1.pose[1] - robot2.pose[1]),
                       abs(robot1.pose[0] - robot2.pose[0]))
        return distance <= self.robot_sensor_range and self._clear_path_between_robots(robot1, robot2)

    def _clear_path_between_robots(self, robot1, robot2):
        coords_of_line = bresenham(start=robot1.pose, end=robot2.pose, world_map=self.maze, occupied_val=self.config['color']['obstacle'])
        Y = [c[0] for c in coords_of_line]
        X = [c[1] for c in coords_of_line]
        points_in_line = self.maze[Y, X]

        if np.any(points_in_line == self.config['color']['obstacle']):
            return False
        else:
            return True

    def _merge_maps(self, robot1, robot2):
        bit_map = np.zeros_like(self.slam_map)
        merge_map = np.ones_like(self.slam_map) * self.config['color']['uncertain']
        for rbt in [robot1, robot2]:
            bit_map = np.bitwise_or(bit_map, rbt.slam_map != self.config['color']['uncertain'])
        idx = np.where(bit_map == 1)
        merge_map[idx] = self.maze[idx]
        robot1.slam_map = merge_map
        robot2.slam_map = merge_map

    def _merge_frontiers_after_communicate(self, robot1, robot2):
        merged_frontiers = merge_frontiers_and_remove_pose(robot1.slam_map, robot1.frontier, robot2.frontier, robot1.pose, robot1.config)
        robot1.frontier = merged_frontiers
        robot2.frontier = merged_frontiers

    def close(self):
        pass

    def _merge_map(self, _map):
        bit_map = np.zeros_like(_map)
        for rbt in self.robots:
            bit_map=np.bitwise_or(bit_map,rbt.slam_map!=self.config['color']['uncertain'])
        idx = np.where(bit_map==1)
        _map[idx]=self.maze[idx]
        return _map

    def _get_state(self):
        state = np.copy(self.slam_map)
        return state

    @property
    def path_length(self):
        length = []
        for _, rbt in enumerate(self.robots):
            length.append(rbt.path_length)
        return length


if __name__ == '__main__':
    env = World()
    maze = np.load('/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl/assets/maps/train/map-57.npy')
    env.maze = maze
    r1 = Robot(1, maze)
    r1.pose = (3,0)
    r2 = Robot(2, maze)
    free_cells = np.argwhere(maze == 0)
    legal_free_cells = set()
    for free_cell in free_cells:
        legal_free_cells.add((free_cell[0], free_cell[1]))
    illegal_free_cells = [(3,0), (4,0), (5,0), (6,0), (5,1), (4,2)]
    for illegal_free_cell in illegal_free_cells:
        legal_free_cells.remove(illegal_free_cell)
    for free_cell in legal_free_cells:
        r2.pose = free_cell
        if env._clear_path_between_robots(r1, r2):
            cv2.rectangle(r1.maze, r2.pose[::-1], r2.pose[::-1], env.config['color']['others'])
    cv2.rectangle(r1.maze, r1.pose[::-1], r1.pose[::-1], env.config['color']['self'])

    plt.imshow(r1.maze)
    plt.show()
