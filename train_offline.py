import argparse
from ast import arg
import sys
import os

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

import pathlib
from datetime import datetime

import numpy as np
import torch

import gym
import d4rl
import d3rlpy

from sklearn.model_selection import train_test_split


from wrappers import Maze2DFixedStartWrapper, Maze2DFirstExitWrapper

custom_dataset = dict(
    point = "maze2d-umaze-v1",
    point_timelimit = "maze2d-umaze-v1_timelimit",
    point_noisy = "maze2d-umaze-v1-noisy",
    point_noisy_timelimit = "maze2d-umaze-v1_timelimit-noisy",
)

def visualize_policy(env, model, num_episodes=10, render=True):
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
            action = model.predict(obs.reshape(1, -1))
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
    std_episode_reward = np.std(total_ep_lengths)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    print(f"-" * 50)
    env.close()
    return total_ep_returns


def pick_d3rlMDPDataset(timelimit, noisy):
    if timelimit:
        if noisy:
            fname = custom_dataset["point_noisy_timelimit"]
            save_name = "noisy_timelimit"
        else:
            fname = custom_dataset["point_timelimit"]
            save_name = "timelimit"
    else:
        if noisy:
            fname = custom_dataset["point_noisy"]
            save_name = "noisy"
        else:
            fname = custom_dataset["point"]
            save_name = "none"
            
    return f"d3rl_{fname}.h5", save_name
   


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, choices=["bcq", "cql"])
    p.add_argument("--train", action="store_true")
    p.add_argument("--render","-r", action="store_true")
    p.add_argument("--timelimit", action="store_true")
    p.add_argument("--noisy", action="store_true")
    args = p.parse_args()
    
    cwd = pathlib.Path(__file__).resolve().parent
    
    log_dir = cwd / "data"
    log_dir_path = pathlib.Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"All Tensorboards and logging are being written inside {log_dir_path}/.")    
   
    # prepare dataset
    dataset_dir = cwd / "generated_dataset" / "d3rl_MDPDataset" 
    d3rl_dataset, save_name = pick_d3rlMDPDataset(args.timelimit, args.noisy)
    dataset = d3rlpy.datasets.MDPDataset.load(dataset_dir / d3rl_dataset)

    
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)
    

    # prepare algorithm
    if args.algo == "bcq":
        model = d3rlpy.algos.BCQ(
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-4,
            use_gpu=True
        )
    elif args.algo == "cql":
        model = d3rlpy.algos.CQL(
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            use_gpu=True
        )
    else:
        raise NotImplementedError()
    
    ISO_TIMESTAMP = "%Y%m%d_%H%M_%S"
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    
    env = Maze2DFixedStartWrapper(gym.make("maze2d-umaze-v1"))
    if not args.timelimit:
        env = Maze2DFirstExitWrapper(env)
        print("Using First Exit Env ...")
    
    # train
    tb_dir = log_dir_path / f"{args.algo.upper()}/{save_name}"
    if args.train:
        model.fit(train_episodes,
                eval_episodes=test_episodes,
                n_epochs=30,
                scorers={
                    'environment': d3rlpy.metrics.evaluate_on_environment(env),
                    'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
                    'td_error': d3rlpy.metrics.td_error_scorer
                },
                save_metrics=True,
                logdir=tb_dir,
                tensorboard_dir=tb_dir,
                save_interval=1,
        )
        model_path = f"./{args.algo.lower()}_{save_name}.pt"
        model.save_model(model_path)
    
    # load entire model parameters.
    model_path = f"./{args.algo.lower()}_{save_name}.pt"
    
    model.build_with_dataset(dataset)
    model.load_model(model_path)

    visualize_policy(env, model, num_episodes=20, render=args.render)

    # ant-v2 speed
    # 