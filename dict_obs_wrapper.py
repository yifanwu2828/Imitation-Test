from collections import OrderedDict
import numpy as np
from gym import ObservationWrapper

class DictObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.desired_goal = np.asarray(self.unwrapped._target)

    def observation(self, observation):
        return self.convertToDictObs(observation)

    def convertToDictObs(self, observation):
        return {
            "achieved_goal": observation[:2],
            "desired_goal": self.desired_goal,
            "observation": observation,
        }

# if __name__ == "__main__":
#     import gym
#     import d4rl
#     from icecream import ic
#     env = gym.make('maze2d-umaze-v1')
#     dict_env = DictObservationWrapper(env)
#     obs = dict_env.reset()
#     for i in range(20):
#         act = dict_env.action_space.sample()
#         obs, *_ = dict_env.step(act)
#         ic(obs)
    
    