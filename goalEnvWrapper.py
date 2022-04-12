import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import numpy as np
# import matplotlib.pyplot as plt

import gym
import d4rl
from gym import GoalEnv
from gym.spaces import Dict, Box


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from dict_obs_wrapper import DictObservationWrapper

# antmaze-umaze-v1

class GoalEnvWrapper(gym.Wrapper, GoalEnv):
    """Downsample the image observation to a square image."""

    def __init__(self, env):
        super().__init__(env)
        self.desired_goal = np.asarray(self.unwrapped._target)
        self.A1 = Polygon([(-2, -2), (-2, 2), (6, 2), (10, -2)])
        self.A2 = Polygon([(6, 2), (10, -2), (10, 10), (6, 6)])
        self.A3 = Polygon([(6, 6), (-2, 6), (-2, 10), (10, 10)])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        add_rew = self.compute_reward(observation[:2], self.desired_goal, info)
        
        return observation, reward + add_rew, done, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.compute_distance(achieved_goal) 
    
    
    def compute_distance(self, achieved_goal):
        x, y = achieved_goal[0], achieved_goal[1]
        point = Point(x, y)
        
        l1 = l2 = l3 = 8
        if self.A1.contains(point):
            distance = l1 - x + l2 + l3
        elif self.A2.contains(point):
            distance = l2 - y + l3
        elif self.A3.contains(point):
            distance = abs(x)
        else:
            raise RuntimeError()
        return distance

        
if __name__ == "__main__":
#     point = Point(10,-2)
#     polygon1 = Polygon([(-2, -2), (-2, 2), (6, 2), (10, -2)])
#     print(polygon1.contains(point))
#     print(polygon1.area)
    
#     A1 = [(-2, -2), (-2, 2), (6, 2), (10, -2)]
#     A2 = [(6, 2), (10, -2), (10, 10), (6, 6)]
#     A3 = [(6, 6), (-2, 6), (-2, 10), (10, 10)]
#     xs, ys = zip(*A1)
#     plt.plot(xs, ys)
#     xs, ys = zip(*A2)
#     plt.plot(xs, ys)
#     xs, ys = zip(*A3)
#     plt.plot(xs, ys)
#     plt.show()
    from icecream import ic
    env = gym.make('maze2d-umaze-v1')
    # dict_env = DictObservationWrapper(env)
    goal_env = GoalEnvWrapper(env)
    obs = goal_env.reset()
    for i in range(20):
        act = goal_env.action_space.sample()
        obs, rew,  *_ = goal_env.step(act)
        ic(rew)