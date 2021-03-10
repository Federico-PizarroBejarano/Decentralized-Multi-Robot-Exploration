import numpy as np
import matplotlib.pyplot as plt

from decentralized_exploration.helpers.plotting import plot_grid

class RobotTeam:
    """
    A class used to represent a team of robots
    
    Instance Attributes
    -------------------
    robots (dict): a dictionary storing the RobotStates of each robot 
        using their robot_ids as keys
    communication_range (float): the maximum range each robot can broadcast its position and map
    blocked_by_obstacles (bool): whether messages are blocked by obstacles

    Public Methods
    --------------
    add_robot(robot): adds a Robot to the team
    explore(world):  given the world the robot is exploring, iteratively explores the area
    """

    def __init__(self, communication_range = float('inf'), blocked_by_obstacles = False):
        self.__robots = {}
        self.__communication_range = communication_range
        self.__blocked_by_obstacles = blocked_by_obstacles


    # Private Methods
    def __generate_message(self, robot_id, world):
        """
        Generates the message that a given robot with robot_id will receive

        Parameters
        ----------
        robot_id (str): the unique id of the robot to receive the message
        world (World): the world which contains the positions of every robot
        """

        message = {}
        robot_position = np.array(world.get_position(robot_id=robot_id))

        for robot in self.__robots.values():
            if (robot.robot_id != robot_id):
                other_robot_position = np.array(world.get_position(robot.robot_id))
                distance = np.linalg.norm(robot_position - other_robot_position) * world.pixel_size

                if distance < self.__communication_range:
                    message[robot.robot_id] = { 
                        'robot_position': other_robot_position,
                        'pixel_map': robot.pixel_map
                    }
        return message


    # Public Methods
    def add_robot(self, robot):
        """
        Adds a new robot to the team if it is not already in the team

        Parameters
        ----------
        robot (Robot): the robot to be added
        """

        if robot.robot_id not in self.__robots:
            self.__robots[robot.robot_id] = robot


    def explore(self, world):
        """
        Given the world the robot is exploring, iteratively explores the area with the whole team

        Parameters
        ----------
        world (World): a World object that the robot will explore
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for robot in self.__robots.values():
            robot.complete_rotation(world=world)
            plot_grid(grid=robot.hex_map, plot=ax, robot_states=world.robot_states)
            plt.pause(0.05)
        
        while self.__robots.values()[0].hex_map.has_rewards():
            for robot in self.__robots.values():
                robot.communicate(message = self.__generate_message(robot_id=robot.robot_id,  world=world))
            
            for robot in self.__robots.values():
                robot.explore_1_timestep(world=world)
            
            plot_grid(grid=robot.hex_map, plot=ax, robot_states=world.robot_states)
            plt.pause(0.05)
            