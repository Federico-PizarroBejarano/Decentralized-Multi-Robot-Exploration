import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import decentralized_exploration.dme_drl.sim_utils as sim_utils
from decentralized_exploration.core.robots.utils.field_of_view import field_of_view, bresenham
from decentralized_exploration.dme_drl.frontier_utils import update_frontier_after_scan
from decentralized_exploration.dme_drl.navigate import AStar, AStarSimple
import os


class Robot():

    def __init__(self, rbt_id, maze, config_path=os.getcwd() + '/assets/config.yaml'):
        with open(config_path) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.id = rbt_id
        self.maze = maze
        self.robot_radius = self.config['robots']['robotRadius']
        self.comm_range = self.config['robots']['commRange']
        self.sync_range = self.config['robots']['syncRange']
        self.sensor_range = 4
        self.probability_of_failed_scan = 0
        self.laser_range = self.config['laser']['range']
        self.laser_fov = self.config['laser']['fov']
        self.laser_resol = self.config['laser']['resolution']
        self.state_size = (self.config['stateSize']['y'], self.config['stateSize']['x'])
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.pose = self._init_pose()
        self.last_map = self.slam_map.copy()
        self.navigator = AStar()
        self.robot_list = None
        self.world = None
        self.path = None
        self.frontier = set()
        self.frontier_by_direction = []

        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect = np.arange(self.laser_range + 1)
        self._radius_vect = radius_vect.reshape(1, radius_vect.shape[0])
        # generate radius vector of [0,1,2,...,laser_range]

        angles_vect = np.arange(-self.laser_fov * 0.5, self.laser_fov * 0.5, step=self.laser_resol)
        self._angles_vect = angles_vect.reshape(angles_vect.shape[0], 1)
        # generate angles vector from -laser_angle/2 to laser_angle

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
        self.pose = self._init_pose()
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = self.slam_map.copy()
        occupied_points, free_points = self._scan()
        self._update_map(occupied_points, free_points)
        self.frontier = update_frontier_after_scan(self.slam_map, self.frontier, free_points, self.pose, self.config)
        self.last_map = np.copy(self.slam_map)
        obs = self.get_obs()
        return obs

    def render(self):
        state = self.get_obs()
        plt.figure(self.id)
        plt.clf()
        plt.imshow(state, cmap='gray')
        plt.pause(0.0001)

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

    def _move_one_step(self, next_point):
        if not self._is_crashed(next_point):
            self.pose = next_point
            map_temp = np.copy(self.slam_map)  # 临时地图，存储原有的slam地图
            occupied_points, free_points = self._scan()
            self._update_map(occupied_points, free_points)
            self.frontier = update_frontier_after_scan(self.slam_map, self.frontier, free_points, self.pose, self.config)
            map_incrmnt = np.count_nonzero(map_temp - self.slam_map)  # map increment
            # self.render()
            return map_incrmnt
        else:
            return -1

    def step(self, action):
        y, x = self.pose
        y_dsti, x_dsti = self.frontier_by_direction[action][0]
        distance_min = np.sqrt((y - y_dsti) ** 2 + (x - x_dsti) ** 2)
        for (y_, x_) in self.frontier_by_direction[action]:
            distance = np.sqrt((y - y_) ** 2 + (x - x_) ** 2)
            if distance < distance_min:
                y_dsti, x_dsti = y_, x_
                distance_min = distance
        self.destination = (y_dsti, x_dsti)
        self.path = self.navigator.navigate(self.maze, self.pose, self.destination)
        counter = 0
        if self.path is None:
            raise Exception('The target point is not accessible')
        else:
            incrmnt_his = []  # map increament list, record the history of it
            for i, point in enumerate(self.path):
                counter += 1
                map_incrmnt = self._move_one_step(point)
                incrmnt_his.append(map_incrmnt)
                if np.sum(incrmnt_his) > 3600:
                    # print('地图增量超过阈值，提前终止探索过程')
                    break
        obs = self.get_obs()
        rwd = self.reward(counter, incrmnt_his)
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
            self.maze == self.config['color']['free']) > 0.95
        info = 'Robot %d has moved to the target point' % (self.id)
        return obs, rwd, done, info

    def move_to_target(self, target):
        """
        move robot to the target position
        :param target: target position, type of np.array
        :return:
        """
        self.destination = target
        if target is None:
            obs = self.get_obs()
            done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
                self.maze == self.config['color']['free']) > 0.95
            info = "No.%d robot fails to move." % self.id
            return obs, 0, done, info
        self.path = self.navigator.navigate(self.maze, self.pose, self.destination)
        if self.path is None:
            raise Exception("The target point is not accessible")
        for point in self.path:
            self._move_one_step(point)
        obs = self.get_obs()
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
            self.maze == self.config['color']['free']) > 0.95
        info = "No.%d robot moves successfully." % self.id
        return obs, None, done, info

    def reward(self, counter, incrmnt_his):
        """reward function"""
        rwd1 = np.sum(incrmnt_his) * self.config['robots']['w1']
        rwd2 = -1. * counter * self.config['robots']['w2']
        rwd = rwd1 + rwd2
        return rwd

    def get_state(self):
        state = self.slam_map.copy()
        for rbt in self.robot_list:
            if rbt.id == self.id:
                cv2.rectangle(state, (rbt.pose[1] - rbt.robot_radius, rbt.pose[0] - rbt.robot_radius),
                              (rbt.pose[1] + rbt.robot_radius, rbt.pose[0] + rbt.robot_radius),
                              color=self.config['color']['self'], thickness=-1)
            else:
                cv2.rectangle(state, (rbt.pose[1] - rbt.robot_radius, rbt.pose[0] - rbt.robot_radius),
                              (rbt.pose[1] + rbt.robot_radius, rbt.pose[0] + rbt.robot_radius),
                              color=self.config['color']['others'], thickness=-1)
        return state.copy()

    def _is_crashed(self, target_point):
        if not sim_utils.within_bound(target_point, self.maze.shape, 0):
            return True
        # 将机器人视为一个质点
        y, x = target_point
        return self.maze[y, x] == self.config['color']['obstacle']

    def get_obs(self):
        """每一个机器人都获取自己观察视野内的本地地图"""
        observation = self.get_state()
        self.get_and_update_frontier_by_direction()
        return observation

    def get_and_update_frontier_by_direction(self):
        """获取前沿，采用地图相减的算法"""
        self.frontier_by_direction = [[] for _ in range(self.config['frontiers']['number'])]

        ry, rx = self.pose[0], self.pose[1]
        for y, x in self.frontier:
            if x - rx != 0:
                tan = (y - ry) / (x - rx)
            else:
                tan = np.sign(y-ry)

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

    def eular_dis(self, point):
        try:
            x, y = point
            return np.linalg.norm([self.pose[0] - x, self.pose[1] - y])
        except TypeError:
            return np.inf

    def select_action(self):
        f_points = [point for front in self.frontier_by_direction for point in front]
        f_dis = list(map(self.eular_dis, f_points))
        return f_points[np.argmin(f_dis)]

    def center_point(self, x):
        return np.mean(x, axis=0)

    def select_action_greedy(self):
        centers = list(map(self.center_point, self.frontier_by_direction))
        cen_dis = list(map(self.eular_dis, centers))
        action = np.argmin(cen_dis)
        f_dis = list(map(self.eular_dis, self.frontier_by_direction[action]))
        return self.frontier_by_direction[action][np.argmin(f_dis)]
        # return self.select_action()

    def select_action_randomly(self):
        action = np.random.randint(8)
        while len(self.frontier_by_direction[action]) == 0:
            action = np.random.randint(8)
        return action

    def select_target_randomly(self):
        f_points = [point for front in self.frontier_by_direction for point in front]
        target = f_points[np.random.randint(0, len(f_points))]
        return target

    @property
    def path_length(self):
        return len(self.path)
