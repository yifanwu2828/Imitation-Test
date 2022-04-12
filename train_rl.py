import sys
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import numpy
import gym
import d4rl
import stable_baselines3 as sb3
from wrappers import Maze2DFixedStartWrapper, Maze2DFirstExitWrapper

def visualize_policy(env, model, num_episodes=10, deterministic=False, render=True):
    """
    Visualize the policy in env
    """
    # Ensure testing on same device
    total_ep_returns = []
    total_ep_lengths = []

    for _ in range(num_episodes):
        obs = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic)
            obs, reward, done, info = env.step(action)
            
            ep_ret += reward
            ep_len += 1

            if render:
                try:
                    env.render()
                except KeyboardInterrupt:
                    sys.exit(0)
            if done:
                total_ep_returns.append(ep_ret)
                total_ep_lengths.append(ep_len)
                obs = env.reset()

    mean_episode_reward = np.mean(total_ep_returns)
    std_episode_reward = np.std(total_ep_returns)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    print(f"-" * 50)
    env.close()
    return total_ep_returns

if __name__ == "__main__":
    
    # env = Maze2DFixedStartWrapper(gym.make('maze2d-umaze-v1'))
    env = gym.make('maze2d-umaze-v1')
    
    model = sb3.SAC(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        learning_starts=500,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
    )
    # model.learn(total_timesteps=1e5, log_interval=2)
    # model.save("maze2d_sac")

    env = Maze2DFixedStartWrapper(env)
    policy = sb3.SAC.load("maze2d_sac") 
    visualize_policy(env, policy, num_episodes=50, deterministic=False, render=True)