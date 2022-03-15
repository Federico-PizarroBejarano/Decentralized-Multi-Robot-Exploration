import numpy as np
import torch as th

from decentralized_exploration.dme_drl.constants import RESET_WORLD_PATH, STEP_WORLD_PATH, TEST_PATH
from decentralized_exploration.dme_drl.eval_robot import EvalRobot
from decentralized_exploration.dme_drl.world import World


class EvalWorld(World):

    def reset(self, test_filename, initial_poses, probability_of_failed_scan, probability_of_failed_communication, failed_communication_interval):
        self.episode += 1
        self.time_step = -1
        self.local_interactions = 0
        self.map_id = TEST_PATH + test_filename
        print('map id： ',self.map_id)
        self.target_points = []
        self.maze = np.load(self.map_id).astype('uint8')
        self.slam_map = np.ones_like(self.maze) * self.config['color']['uncertain']
        self.last_map = np.copy(self.slam_map)
        self.track_map = np.copy(self.maze)
        self.data_transmitted = 0
        self.probability_of_failed_communication = probability_of_failed_communication
        self.robots = [EvalRobot(i, np.copy(self.maze)) for i in range(self.number)]
        self.failed_communication_interval = failed_communication_interval

        for id, robot in enumerate(self.robots):
            robot.robots = self.robots
            robot.world = self
            robot.reset(np.copy(self.maze), initial_poses[id], probability_of_failed_scan)

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

    def communicate(self, robot1, robot2):
        if self._can_communicate():
            robot1.render(robot1.render_path + 'r{}_s{}_pre_merge_with_r{}_former'.format(robot1.id, robot1.counter, robot2.id))
            robot2.render(robot1.render_path + 'r{}_s{}_pre_merge_with_r{}_latter'.format(robot1.id, robot1.counter, robot2.id))

            self._merge_maps(robot1, robot2)
            self._merge_frontiers_after_communicate(robot1, robot2)

            robot1.render(robot1.render_path + 'r{}_s{}_pro_merge_with_r{}_former'.format(robot1.id, robot1.counter, robot2.id))
            robot2.render(robot1.render_path + 'r{}_s{}_pro_merge_with_r{}_latter'.format(robot1.id, robot1.counter, robot2.id))
        else:
            robot1.comm_dropout_steps = self.failed_communication_interval
            robot2.comm_dropout_steps = self.failed_communication_interval
