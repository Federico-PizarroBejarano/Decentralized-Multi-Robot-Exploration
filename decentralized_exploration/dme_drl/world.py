import os
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml

from decentralized_exploration.dme_drl.frontier_utils import merge_frontiers_and_remove_pose, remove_pose_from_frontier, \
    cleanup_frontier
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
            self.ax = self.fig.add_subplot(111)

    def record_poses(self, robot1, robot2):
        robot1.poses[:, 2 * robot2.id] = robot2.pose[0]
        robot1.poses[:, 2 * robot2.id + 1] = robot2.pose[1]

        robot2.poses[:, 2 * robot1.id] = robot1.pose[0]
        robot2.poses[:, 2 * robot1.id + 1] = robot1.pose[1]

        robot1.seen_robots.add(robot2.id)
        robot2.seen_robots.add(robot1.id)

    def communicate(self, robot1, robot2):
        if self._can_communicate():
            robot1.render(STEP_WORLD_PATH + 'step_robot_{}_{}_before_comm_t{}'.format(robot1.id, robot2.id, self.time_step))
            robot2.render(STEP_WORLD_PATH + 'step_robot_{}_{}_before_comm_t{}'.format(robot2.id, robot1.id, self.time_step))

            self._merge_maps(robot1, robot2)
            self._merge_frontiers_after_communicate(robot1, robot2)

            robot1.render(STEP_WORLD_PATH + 'step_robot_{}_{}_after_comm_t{}'.format(robot1.id, robot2.id, self.time_step))
            robot2.render(STEP_WORLD_PATH + 'step_robot_{}_{}_after_comm_t{}'.format(robot2.id, robot1.id, self.time_step))


    def reset(self,random=True):
        self.episode += 1
        self.time_step = -1
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
        if manual_check:
            self.render(RESET_WORLD_PATH + 'reset_world_before_merge_and_comm')
        for rbt in self.robots:
            rbt.robots=self.robots
            rbt.world = self
            rbt.reset(np.copy(self.maze))
        self.slam_map = self._merge_map(self.slam_map)
        if manual_check:
            self.render(RESET_WORLD_PATH + 'reset_world_after_merge')
        obs_n = []
        pose_n = []

        for id1, robot1 in enumerate(self.robots):
            for id2, robot2 in enumerate(self.robots):
                if id1 < id2:
                    if self.in_range(robot1, robot2):
                        self.record_poses(robot1, robot2)
                        self.communicate(robot1, robot2)
            obs_n.append(robot1.get_obs())
            pose_n.append(robot1.get_poses())

        return obs_n,pose_n

    def seed(self, seed=None):
        pass

    def _track(self):
        for i,rbt in enumerate(self.robots):
            for p in rbt.path:
                color = np.zeros(3)
                if i%2 == 0 :
                    color[0] = 255
                    cv2.circle(self.track_map, (p[1], p[0]), 2, color, -1)
                else:
                    color[2] = 255
                    cv2.rectangle(self.track_map, (p[1], p[0]), (p[1]+1, p[0]+1), color, -1)

    def render(self, fname=None):
        # update the global frontier
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
            np.save(fname, self.slam_map)
        else:
            plt.pause(0.5)

    def step(self, action_n):
        # action_n: 0~7
        self.time_step += 1
        obs_n = []
        rwd_n = []
        info_n = []
        pose_n = []

        if manual_check:
            step_world_path = STEP_WORLD_PATH + 'e{}_t{}/'.format(self.episode, self.time_step)
            step_robot_path = STEP_ROBOT_PATH + 'e{}_t{}/'.format(self.episode, self.time_step)
            os.makedirs(step_world_path, exist_ok=True)
            os.makedirs(step_robot_path, exist_ok=True)
            self.render(step_world_path + 'before_merge_and_comm')

        for id1, rbt in enumerate(self.robots):
            if action_n[id1] == -1:
                # NOOP
                obs = rbt.get_obs()
                rwd = -2
                info = 'NOOP'
            else:
                if manual_check:
                    obs, rwd, done, info = rbt.step(action_n[id1], step_robot_path)
                else:
                    obs, rwd, done, info = rbt.step(action_n[id1])
            obs_n.append(obs)
            rwd_n.append(rwd)
            info_n.append(info)
            pose = np.ones((1, self.number * 2)) * (-1)
            pose[:, 2 * id1] = rbt.pose[0]
            pose[:, 2 * id1 + 1] = rbt.pose[1]
            pose_n.append(pose)
        self.slam_map = self._merge_map(self.slam_map)
        if manual_check:
            self.render(step_world_path + 'after_merge')
        for id1, robot1 in enumerate(self.robots):
            for id2, robot2 in enumerate(self.robots):
                if id1 < id2:
                    # layers communication
                    if self.in_range(robot1, robot2):
                        # exchange position information
                        pose_n[id1][:, 2 * id2] = robot2.pose[0]
                        pose_n[id1][:, 2 * id2 + 1] = robot2.pose[1]

                        self.communicate(robot1, robot2)
            robot1.seen_robots = set() # clear seen robots

        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95
        return obs_n,rwd_n,done,info_n,pose_n

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
