import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml

from decentralized_exploration.dme_drl.frontier_utils import merge_frontiers_and_remove_pose, remove_pose_from_frontier, \
    cleanup_frontier
from decentralized_exploration.dme_drl.robot import Robot
from decentralized_exploration.core.robots.utils.field_of_view import bresenham
from decentralized_exploration.dme_drl.constants import render_world, PROJECT_PATH, CONFIG_PATH


class World(gym.Env):
    def __init__(self, number=None):
        with open(CONFIG_PATH) as stream:
            self.config = yaml.load(stream, Loader=yaml.SafeLoader)
        self.map_id_set_train = np.loadtxt(PROJECT_PATH + self.config['map_id_train_set'], str)
        # self.map_id_set_eval = np.loadtxt(os.getcwd()+self.config['map_id_eval_set'],str)
        if number is None:
            self.number = self.config['robots']['number']
        else:
            self.number = number
        # parameters will be set in reset func
        self.map_id = None
        self.target_points = []
        self.maze = np.zeros([1, 1])
        self.slam_map = np.zeros_like(self.maze)
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.zeros_like(self.slam_map)
        self.data_transmitted = 0
        self.robots = []
        self.robot_sensor_range = self.config['robots']['sensorRange']
        self.probability_of_failed_communication = self.config['robots']['probabilityOfFailedCommunication']
        self.frontier = set()
        if render_world:
            self.fig = plt.figure('global')
            self.ax = self.fig.add_subplot(111)
        self.reset()

    def reset(self,random=True):
        if random:
            self.map_id = PROJECT_PATH + np.random.choice(self.map_id_set_train)
        # else:
            # self.map_id = self.map_id_set_eval[0]
            # self.map_id_set_eval = np.delete(self.map_id_set_eval,0)
        print('map id： ',self.map_id)
        self.target_points = []
        self.maze = np.load(self.map_id).astype('uint8')
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.copy(self.maze)
        self.data_transmitted = 0
        self.robots = [Robot(i, np.copy(self.maze)) for i in range(self.number)]
        for rbt in self.robots:
            rbt.robot_list=self.robots
            rbt.world = self
            rbt.reset(np.copy(self.maze))
        self._merge_map()
        obs_n = []
        pose_n = []
        for i,rbt in enumerate(self.robots):
            obs_n.append(rbt.get_obs())
            pose = np.ones((1, self.number * 2)) * (-1)
            pose[:,2*i] = rbt.pose[0]
            pose[:,2*i+1] = rbt.pose[1]
            pose_n.append(pose)

        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if not i==j:
                    distance = max(abs(robot1.pose[1] - robot2.pose[1]),
                                   abs(robot1.pose[0] - robot2.pose[0]))
                    if self._is_in_range(distance, robot1, robot2):
                        # exchange position information
                        pose_n[i][:,2*j] = robot2.pose[0]
                        pose_n[i][:,2*j+1] = robot2.pose[1]
                        if self._can_communicate():
                            self._communicate(robot1, robot2)
                            self._merge_frontiers_after_communicate(robot1, robot2)
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

    def render(self):
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
            self.ax.scatter(robot.pose[1], robot.pose[0], color='y', marker='s', alpha=1, s=140)
            self.ax.text(robot.pose[1], robot.pose[0], s=robot.id, ha='center', va='center', size=8)

        for node in global_frontier:
            self.ax.text(node[1], node[0], 'F', ha='center', va='center', size=8)

        self.ax.set_xlim(-0.5, 19.5)
        self.ax.set_ylim(-0.5, 19.5)

        self.ax.invert_yaxis()

        plt.pause(0.5)

    def step(self, action_n):
        # action_n: 0~7
        obs_n = []
        rwd_n = []
        info_n = []
        pose_n = []
        for i, rbt in enumerate(self.robots):
            if action_n[i] == -1:
                # NOOP
                obs = rbt.get_obs()
                rwd = -2
                info = 'NOOP'
            else:
                obs, rwd, done, info = rbt.step(action_n[i])
            obs_n.append(obs)
            rwd_n.append(rwd)
            info_n.append(info)
            pose = np.ones((1, self.number * 2)) * (-1)
            pose[:, 2 * i] = rbt.pose[0]
            pose[:, 2 * i + 1] = rbt.pose[1]
            pose_n.append(pose)
        self._merge_map()
        for i, robot1 in enumerate(self.robots):
            for j, robot2 in enumerate(self.robots):
                if not i == j:
                    distance = max(abs(robot1.pose[1] - robot1.pose[1]),
                                   abs(robot1.pose[0] - robot1.pose[0]))
                    # layers communication
                    if self._is_in_range(distance, robot1, robot2):
                        # exchange position information
                        pose_n[i][:, 2 * j] = robot2.pose[0]
                        pose_n[i][:, 2 * j + 1] = robot2.pose[1]
                        if self._can_communicate():
                            # exchange complete information
                            self._communicate(robot1, robot2)
                            self._merge_frontiers_after_communicate(robot1, robot2)

        # self.render()
        self._track()
        done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(self.maze == self.config['color']['free']) > 0.95
        #if done:
            #self.track()
        return obs_n,rwd_n,done,info_n,pose_n

    def _can_communicate(self):
        return np.random.randint(100) > self.probability_of_failed_communication

    def _is_in_range(self, distance, robot1, robot2):
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

    def _communicate(self, rbt1, rbt2):
        bit_map = np.zeros_like(self.slam_map)
        merge_map = np.ones_like(self.slam_map) * self.config['color']['uncertain']
        for rbt in [rbt1, rbt2]:
            bit_map = np.bitwise_or(bit_map, rbt.slam_map != self.config['color']['uncertain'])
        idx = np.where(bit_map == 1)
        merge_map[idx] = self.maze[idx]
        rbt1.slam_map = merge_map
        rbt2.slam_map = merge_map
        return

    def _merge_frontiers_after_communicate(self, rbt1, rbt2):
        merged_frontiers = merge_frontiers_and_remove_pose(rbt1.slam_map, rbt1.frontier, rbt2.frontier, rbt1.pose, rbt1.config)
        rbt1.frontier = merged_frontiers

    def close(self):
        pass

    def _merge_map(self):
        bit_map = np.zeros_like(self.slam_map)
        for rbt in self.robots:
            bit_map=np.bitwise_or(bit_map,rbt.slam_map!=self.config['color']['uncertain'])
        idx = np.where(bit_map==1)
        self.slam_map[idx]=self.maze[idx]
        return

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
