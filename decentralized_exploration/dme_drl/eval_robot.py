import os

import numpy as np

from decentralized_exploration.dme_drl.constants import RESET_ROBOT_PATH, manual_check, \
    STEP_ROBOT_PATH
from decentralized_exploration.dme_drl.frontier_utils import update_frontier_and_remove_pose
from decentralized_exploration.dme_drl.robot import Robot

ID = 0

class EvalRobot(Robot):

    def get_pose(self):
        return np.array(self.pose)

    def reset(self, maze, pose, probability_of_failed_scan):
        self.maze = maze
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.pose = pose
        self.poses = np.ones((1, self.number * 2)) * (-1)
        self.poses[:, 2 * self.id] = self.pose[0]
        self.poses[:, 2 * self.id + 1] = self.pose[1]
        self.episode += 1
        self.time_step = -1
        self.sub_time_step = -1
        self.distance = 0
        self.distance_travelled_history = []
        self.pose_history = [self.get_pose()]
        self.probability_of_failed_scan = probability_of_failed_scan
        self.probability_of_failed_action = 10 - 1
        self.comm_dropout_steps = 0

        self.render(RESET_ROBOT_PATH + 'r{}_e{}_t{}_pre_reset'.format(self.id, self.episode, self.time_step))
        self.map_history = [self.get_obs()]

        prev_world_map = self.world.slam_map.copy()

        occupied_points, free_points = self._scan()
        self._update_map(occupied_points, free_points)
        self.frontier = update_frontier_and_remove_pose(self.slam_map, self.frontier, self.pose, self.config)
        self.world.slam_map = self.world._merge_map(self.world.slam_map)

        new_world_map = self.world.slam_map.copy()
        area_explored = np.count_nonzero(new_world_map - prev_world_map)
        self.area_explored_history = [area_explored]

        self.map_history.append(self.get_obs())
        self.pose_history.append(self.get_pose())
        self.render(RESET_ROBOT_PATH + 'r{}_e{}_t{}_pro_reset'.format(self.id, self.episode, self.time_step))

        self.last_map = self.slam_map.copy()

    def next_possible_action(self, action):
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

    def _can_move(self):
        return np.random.randint(100) > self.probability_of_failed_action

    def step(self, maddpg, obs_history, pose):
        self.time_step += 1

        self.render_path = STEP_ROBOT_PATH + 'e{}_t{}/'.format(self.episode, self.time_step)
        if manual_check:
            os.makedirs(self.render_path, exist_ok=True)

        y, x = self.pose

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
                done = np.array_equal(self.world.slam_map == self.config['color']['free'],
                                      self.world.maze == self.config['color']['free'])

                if self._can_move():
                    next_point = point
                else:
                    legal_actions = []
                    for i in range(8):
                        next_possible_point = self.next_possible_action(i)
                        if self._is_legal(next_possible_point):
                            legal_actions.append(next_possible_point)
                    next_point = legal_actions[np.random.randint(len(legal_actions))]


                if self.in_vicinity_and_not_yet_seen() or\
                        self.sub_time_step == 300 or\
                        not self._is_legal(next_point) or \
                        done: # enforce sub_time_step limit
                    break


                self.sub_time_step += 1
                self.counter += 1
                self.comm_dropout_steps = max(0, self.comm_dropout_steps - 1)

                distance_travelled = ((point[0] - self.pose[0])**2 + (point[1] - self.pose[1])**2)**0.5
                self.distance += distance_travelled
                self.distance_travelled_history.append(distance_travelled)

                prev_world_map = self.world.slam_map.copy()
                map_increment = self._move_one_step(next_point, action)
                self.world.slam_map = self.world._merge_map(self.world.slam_map)
                new_world_map = self.world.slam_map.copy()
                area_explored = np.count_nonzero(new_world_map-prev_world_map)

                self.pose_history.append(self.get_pose())
                self.map_history.append(self.get_obs())

                if self.sub_time_step == 0:
                    self.area_explored_history[0] + area_explored
                else:
                    self.area_explored_history.append(area_explored)


                increment_his.append(map_increment)

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

    def in_vicinity_and_not_yet_seen(self):
        flag = False
        for robot in self.robots:
            if self.id != robot.id and self.world.in_range(self, robot) and robot.id not in self.seen_robots:
                self.world.record_poses(self, robot)
                if self.comm_dropout_steps == 0 and robot.comm_dropout_steps == 0:
                    self.world.communicate(self, robot)
                flag = True
        return flag
