import os
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import decentralized_exploration.dme_drl.sim_utils as sim_utils
from decentralized_exploration.core.robots.utils.field_of_view import bresenham
from decentralized_exploration.dme_drl.frontier_utils import update_frontier_and_remove_pose
from decentralized_exploration.dme_drl.constants import CONFIG_PATH, render_robot_map, manual_check, ACTION_TO_NAME, \
	RESET_ROBOT_DIR, ID_TO_COLOR, STEP_ROBOT_DIR

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
		self.probability_of_failed_scan = self.config['robots']['probabilityOfFailedScan']
		self.probability_of_failed_action = self.config['robots']['probabilityOfFailedAction']
		self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
		self.pose = self._init_pose()
		self.navigator = AStar()
		self.robot_list = None
		self.world = None
		self.path = None
		self.destination = None
		self.frontier = set()
		self.frontier_by_direction = []
		if manual_check:# and self.id == ID:
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
		self.pose = self._init_pose()
		self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']

		self.render(RESET_ROBOT_DIR + 'reset_robot_{}_before_scan'.format(self.id))

		occupied_points, free_points = self._scan()
		self._update_map(occupied_points, free_points)
		self.frontier = update_frontier_and_remove_pose(self.slam_map, self.frontier, self.pose, self.config)

		self.render(RESET_ROBOT_DIR + 'reset_robot_{}_after_scan'.format(self.id))

		obs = self.get_obs()
		return obs

	def render(self, fname):
		if manual_check:
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

			if render_robot_map:
				plt.pause(0.5)
			else:
				self.fig.savefig(fname)

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
		return 0 <= point[0] < self.slam_map.shape[0] and\
		       0 <= point[1] < self.slam_map.shape[1]

	def _is_free_space(self, point):
		for robot in self.robot_list:
			if robot.pose == point:
				return False

		return self.slam_map[point] == self.config['color']['free']

	def _is_legal(self, point):
		return self._is_in_bounds(point) and self._is_free_space(point)

	def _move_one_step_and_scan(self, next_point, step_path=None):
		if self._is_legal(next_point):
			last_map = self.slam_map.copy()

			self.pose = next_point

			occupied_points, free_points = self._scan()
			self._update_map(occupied_points, free_points)
			self.frontier = update_frontier_and_remove_pose(self.slam_map, self.frontier, self.pose, self.config)

			map_incrmnt = np.count_nonzero(last_map - self.slam_map)  # map increment
			return map_incrmnt
		else:
			return -1

	def _can_move(self):
		return np.random.randint(100) > self.probability_of_failed_action

	def next_point(self, action):
		if action == 0:
			return self.pose[0] + 1, self.pose[1]
		elif action == 1:
			return self.pose[0] + 1, self.pose[1] + 1
		elif action == 2:
			return self.pose[0], self.pose[1] + 1
		elif action == 3:
			return self.pose[0] - 1, self.pose[1] + 1
		elif action == 4:
			return self.pose[0] - 1, self.pose[1]
		elif action == 5:
			return self.pose[0] - 1, self.pose[1] - 1
		elif action == 6:
			return self.pose[0], self.pose[1] - 1
		elif action == 7:
			return self.pose[0] + 1, self.pose[1] - 1


	def step(self, action, step_robot_path=None):
		step_robot_path += 'robot_{}_{}'.format(self.id, ACTION_TO_NAME[action])

		increment_his = []

		if self._can_move():
			next_point = self.next_point(action)
		else:
			legal_actions = []
			for i in range(8):
				possible_next_point = self.next_point(i)
				if self._is_legal(possible_next_point):
					legal_actions.append(possible_next_point)
			next_point = legal_actions[np.random.randint(len(legal_actions))]

		self.render(step_robot_path + '-before_step')
		map_increment = self._move_one_step_and_scan(next_point, step_robot_path)
		self.render(step_robot_path + '-after_step')

		increment_his.append(map_increment)

		obs = self.get_obs()
		rwd = self.reward(1, increment_his)
		done = np.sum(self.slam_map == self.config['color']['free']) / np.sum(
			self.maze == self.config['color']['free']) > 0.95
		info = 'Robot %d has moved to the target point' % (self.id)
		return obs, rwd, done, info



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

	def get_obs(self):
		"""每一个机器人都获取自己观察视野内的本地地图"""
		return self.slam_map
		# observation = self.get_state()
		# return observation

	@property
	def path_length(self):
		return len(self.path)

if __name__ == "__main__":
	maze = np.load('/Users/richardren/VisualStudioCodeProjects/Decentralized-Multi-Robot-Exploration/decentralized_exploration/dme_drl/assets/maps/train/map-57.npy')
	robot = Robot(2, maze)
	robot.pose = 6,4
	robot.robot_list = [robot]
	occ_points, free_points = robot._scan()
	robot._update_map(occ_points, free_points)
	plt.imshow(robot.get_state())
	plt.show()
