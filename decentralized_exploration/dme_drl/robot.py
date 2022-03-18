import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

import torch as th
from torch.distributions import categorical
from decentralized_exploration.dme_drl.sim_utils import onehot_from_action

from decentralized_exploration.core.robots.utils.field_of_view import bresenham
from decentralized_exploration.dme_drl.frontier_utils import update_frontier_and_remove_poses
from decentralized_exploration.dme_drl.constants import CONFIG_PATH, render_robot_map, RESET_ROBOT_PATH, manual_check, \
    ID_TO_COLOR, STEP_ROBOT_PATH, ACTION_TO_NAME
from decentralized_exploration.dme_drl.navigate import AStar

ID = 0

class Robot():

    def __init__(self, rbt_id, maze):
        with open(CONFIG_PATH) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.id = rbt_id
        self.maze = maze
        self.robot_radius = self.config['robots']['robotRadius']
        self.sensor_range = self.config['robots']['sensorRange']
        self.probability_of_failed_scan = self.config['robots']['probabilityOfFailedScan']
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.pose = None
        self.poses = None
        self.number = self.config['robots']['number']
        self.last_map = self.slam_map.copy()
        self.navigator = AStar()
        self.robots = None
        self.world = None
        self.path = None
        self.destination = None
        self.frontier = set()
        self.frontier_by_direction = []
        self.seen_robots = set()
        self.episode = -1
        self.time_step = -1
        self.sub_time_step = -1
        self.distance = 0
        if render_robot_map or manual_check:
            self.fig = plt.figure('robot ' + str(self.id))
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)

    def _init_pose(self):
        if self.config['robots']['resetRandomPose'] == 1:
            h, w = self.maze.shape
            y_min, y_max = int(0.1 * h), int(0.8 * h)
            x_min, x_max = int(0.1 * w), int(0.8 * w)
            y = np.random.randint(y_min, y_max)
            x = np.random.randint(x_min, x_max)
            while self.maze[y, x] == self.config['color']['obstacle']:
                y = np.random.randint(y_min, y_max)
                x = np.random.randint(x_min, x_max)
            return y, x
        else:
            return self.config['robots']['startPose']['y'], self.config['robots']['startPose']['x']

    def reset(self, maze):
        self.maze = maze
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.pose = self._init_pose()
        self.poses = np.ones((1, self.number * 2)) * (-1)
        self.poses[:, 2 * self.id] = self.pose[0]
        self.poses[:, 2 * self.id + 1] = self.pose[1]
        self.episode += 1
        self.time_step = -1
        self.sub_time_step = -1
        self.distance = 0

        self.render(RESET_ROBOT_PATH + 'r{}_e{}_t{}_pre_reset'.format(self.id, self.episode, self.time_step))

        occupied_points, free_points = self._scan()
        self._update_map(occupied_points, free_points)
        self.frontier = update_frontier_and_remove_poses(self.slam_map, self.frontier, self.poses, self.config)

        self.render(RESET_ROBOT_PATH + 'r{}_e{}_t{}_pro_reset'.format(self.id, self.episode, self.time_step))

        self.last_map = self.slam_map.copy()

    def render(self, fname):
        if manual_check or render_robot_map:
            self.ax.cla()
            self.ax.set_aspect('equal')

            # plot the terrain
            for y in range(self.slam_map.shape[0]):
                for x in range(self.slam_map.shape[1]):
                    val = self.slam_map[y, x]
                    if val == self.config['color']['uncertain']:
                        c = 'gray'
                    if val == self.config['color']['obstacle']:
                        c = 'black'
                    if val == self.config['color']['free']:
                        c = 'white'

                    self.ax.scatter(x, y, color=c, alpha=0.75, marker='s', s=140)

            # plot the destination
            if self.destination:
                self.ax.scatter(self.destination[1], self.destination[0], color='y', marker='s', alpha=1, s=140)

            # plot the robot
            self.ax.scatter(self.pose[1], self.pose[0], color=ID_TO_COLOR[self.id], marker='s', alpha=1, s=140)
            self.ax.text(self.pose[1], self.pose[0], s=self.id, ha='center', va='center', size=8)


            for node in self.frontier:
                self.ax.text(node[1], node[0], 'F', ha='center', va='center', size=8)

            self.ax.set_xlim(-0.5, 19.5)
            self.ax.set_ylim(-0.5, 19.5)

            if manual_check:
                self.fig.savefig(fname)
                # np.save(fname, self.slam_map)
            else:
                plt.pause(0.5)

    def _scan(self):
        world_size = self.maze.shape
        radius = self.sensor_range
        y, x = self.pose

        all_free_points = set()
        all_occupied_points = set()

        for yi in (max(y - radius, 0), min(y + radius, world_size[0] - 1)):
            for xi in range(max(x - radius, 0), min(x + radius, world_size[1] - 1) + 1):
                all_points = bresenham(start=self.pose, end=[yi, xi], world_map=self.maze, occupied_val=self.config['color']['obstacle'])
                all_free_points = all_free_points.union(set(all_points[:-1]))
                if self.maze[all_points[-1][0], all_points[-1][1]] == self.config['color']['obstacle']:
                    all_occupied_points.add(all_points[-1])
                else:
                    all_free_points.add(all_points[-1])

        for yi in range(max(y - radius, 0), min(y + radius, world_size[0] - 1) + 1):
            for xi in (max(x - radius, 0), min(x + radius, world_size[1] - 1)):
                all_points = bresenham(start=self.pose, end=[yi, xi], world_map=self.maze, occupied_val=self.config['color']['obstacle'])
                all_free_points = all_free_points.union(set(all_points[:-1]))
                if self.maze[all_points[-1][0], all_points[-1][1]] == self.config['color']['obstacle']:
                        all_occupied_points.add(all_points[-1])
                else:
                    all_free_points.add(all_points[-1])

        all_occupied = list(all_occupied_points)
        all_free = list(all_free_points)

        keep_occ = [np.random.randint(100) > self.probability_of_failed_scan for i in range(len(all_occupied))]
        keep_free = [np.random.randint(100) > self.probability_of_failed_scan for i in range(len(all_free))]
        all_occupied_points = set([all_occupied[p] for p in range(len(all_occupied)) if keep_occ[p]])
        all_free_points = set([all_free[p] for p in range(len(all_free)) if keep_free[p]])

        return all_occupied_points, all_free_points

    def _update_map(self, occupied_points, free_points):
        occupied_points = [p for p in occupied_points if self.slam_map[p[0], p[1]] == self.config['color']['uncertain']]
        free_points = [p for p in free_points if self.slam_map[p[0], p[1]] == self.config['color']['uncertain']]

        occ_rows, occ_cols = [p[0] for p in occupied_points], [p[1] for p in occupied_points]
        free_rows, free_cols = [p[0] for p in free_points], [p[1] for p in free_points]

        self.slam_map[occ_rows, occ_cols] = self.config['color']['obstacle']
        self.slam_map[free_rows, free_cols] = self.config['color']['free']

        # update the map
        return np.copy(self.slam_map)

    def _is_in_bounds(self, point):
        return 0 <= point[0] < self.slam_map.shape[0] and \
               0 <= point[1] < self.slam_map.shape[1]

    def _is_free_space(self, point):
        for robot in self.robots:
            if robot.pose == point:
                return False

        return self.slam_map[point] == self.config['color']['free']

    def _is_one_step(self, point):
        distance = max(abs(self.pose[1] - point[1]),
                       abs(self.pose[0] - point[0]))
        return 0 <= distance <= 1

    def _is_legal(self, point):
        return self._is_in_bounds(point) and self._is_free_space(point) and self._is_one_step(point)

    def _move_one_step(self, next_point, action):
        if self._is_legal(next_point):
            self.render(self.render_path + 'r{}_s{}_pre_step_{}'.format(self.id, self.counter, ACTION_TO_NAME[action]))
            self.pose = next_point

            map_temp = np.copy(self.slam_map)  # 临时地图，存储原有的slam地图

            occupied_points, free_points = self._scan()
            self._update_map(occupied_points, free_points)
            self.frontier = update_frontier_and_remove_poses(self.slam_map, self.frontier, self.poses, self.config)

            map_increment = np.count_nonzero(map_temp - self.slam_map)  # map increment
            self.render(self.render_path + 'r{}_s{}_pro_step_{}'.format(self.id, self.counter, ACTION_TO_NAME[action]))
            return map_increment
        else:
            return -1

    def in_vicinity_and_not_yet_seen(self):
        flag = False
        for robot in self.robots:
            if self.id != robot.id and self.world.in_range(self, robot) and robot.id not in self.seen_robots:
                self.world.record_poses(self, robot)
                self.world.communicate(self, robot)
                flag = True
        return flag

    def select_action(self, maddpg, obs_history, pose):
        obs_history[self.id, 0:20, :] = th.from_numpy(self.get_obs())
        pose[self.id] = th.from_numpy(self.get_poses())

        action_probs = maddpg.select_action(obs_history, pose).data.cpu()
        action_probs_valid = np.copy(action_probs)
        action = []

        for direction, frontier in enumerate(self.get_and_update_frontier_by_direction()):
            if len(frontier) == 0:
                action_probs_valid[self.id][direction] = 0

        if np.array_equal(action_probs_valid[self.id], np.zeros_like(action_probs_valid[self.id])):
            return None

        else:
            act = categorical.Categorical(probs=th.tensor(action_probs_valid[self.id]))
            sample_act = act.sample()
            action.append(sample_act)

        self.action = th.tensor(onehot_from_action(action))
        acts = np.argmax(self.action, axis=1)[0]

        if len(self.frontier_by_direction[acts]) == 0:
            return -1
        else:
            return acts


    def step(self, maddpg, obs_history, pose):
        self.time_step += 1

        self.render_path = STEP_ROBOT_PATH + 'e{}_t{}/'.format(self.episode, self.time_step)
        if manual_check:
            os.makedirs(self.render_path, exist_ok=True)

        y, x = self.pose

        self.frontier = update_frontier_and_remove_poses(self.slam_map, self.frontier, self.poses, self.config)
        self.frontier_by_direction = self.get_and_update_frontier_by_direction()
        action = self.select_action(maddpg, obs_history, pose)

        if action is None: # empty frontier
            return None, 'empty frontier'
        elif action == -1:
            return -2, None

        y_dsti, x_dsti = self.frontier_by_direction[action][0]
        distance_min = np.sqrt((y - y_dsti) ** 2 + (x - x_dsti) ** 2)
        for (y_, x_) in self.frontier_by_direction[action]:
            distance = np.sqrt((y - y_) ** 2 + (x - x_) ** 2)
            if distance < distance_min:
                y_dsti, x_dsti = y_, x_
                distance_min = distance
        self.destination = (y_dsti, x_dsti)
        self.path = self.navigator.navigate(self.maze, self.pose, self.destination)
        self.counter = 0

        if self.path is None:
            raise Exception('The target point is not accessible')
        else:
            increment_his = []  # map increment list, record the history of it
            for i, point in enumerate(self.path):
                if self.in_vicinity_and_not_yet_seen():
                    break

                self.counter += 1
                self.distance += ((point[0] - self.pose[0])**2 + (point[1] - self.pose[1])**2)**0.5

                map_increment = self._move_one_step(point, action)
                increment_his.append(map_increment)

        self.sub_time_step += self.counter

        self.destination = None
        self.last_map = self.slam_map.copy()
        rwd = self.reward(self.counter, increment_his)
        info = 'Robot %d has moved to the target point' % (self.id)

        # cleanup
        self.seen_robots.clear() # clear seen self
        self.poses = np.ones((1, self.number * 2)) * (-1)
        self.poses[:, 2 * self.id] = self.pose[0]
        self.poses[:, 2 * self.id + 1] = self.pose[1]

        return rwd, info



    def reward(self, counter, incrmnt_his):
        """reward function"""
        rwd1 = np.sum(incrmnt_his) * self.config['robots']['w1']
        rwd2 = -1. * counter * self.config['robots']['w2']
        rwd = rwd1 + rwd2
        return rwd

    def get_obs(self):
        """每一个机器人都获取自己观察视野内的本地地图"""
        return self.slam_map.copy()

    def get_poses(self):
        return self.poses.copy()

    def get_action(self):
        return self.action.clone().detach()


    def get_and_update_frontier_by_direction(self):
        """获取前沿，采用地图相减的算法"""
        self.frontier_by_direction = [[] for _ in range(self.config['frontiers']['number'])]

        ry, rx = self.pose[0], self.pose[1]
        for y, x in self.frontier:
            if x - rx != 0:
                tan = (y - ry) / (x - rx)
            else:
                tan = ry - y # since else clause is invoked

            if x > rx:
                if 1 <= tan:
                    self.frontier_by_direction[0].append((y, x))
                elif 0 <= tan < 1:
                    self.frontier_by_direction[1].append((y, x))
                elif -1 <= tan < 0:
                    self.frontier_by_direction[2].append((y, x))
                elif tan < -1:
                    self.frontier_by_direction[3].append((y, x))
            else:
                if 1 <= tan:
                    self.frontier_by_direction[4].append((y, x))
                elif 0 <= tan < 1:
                    self.frontier_by_direction[5].append((y, x))
                elif -1 <= tan < 0:
                    self.frontier_by_direction[6].append((y, x))
                elif tan < -1:
                    self.frontier_by_direction[7].append((y, x))

        if len(self.frontier_by_direction) > 0:
            return np.copy(self.frontier_by_direction)
        else:
            raise Exception('Exception: None Contour!')

    @property
    def path_length(self):
        return len(self.path)

if __name__ == "__main__":
    maze = np.load('/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl/assets/maps/train/map-57.npy')
    robot = Robot(2, maze)
    robot.pose = 6,4
    robot.robots = [robot]
    occ_points, free_points = robot._scan()
    robot._update_map(occ_points, free_points)
    plt.imshow(robot.get_state())
    plt.show()
