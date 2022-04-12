import sys
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import argparse
import pathlib
import pickle
import tempfile
from datetime import datetime
from typing import Iterable, Optional, Type


import yaml

import gym
import d4rl # Import required to register environments

import numpy as np
import torch as th
from torch import  nn
from torch.nn.utils import spectral_norm

from tqdm import trange

from imitation.util import networks

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
# dummy_vec_env
from stable_baselines3.common import preprocessing
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util
from imitation.rewards.reward_nets import RewardNet

from wrappers import Maze2DFixedStartWrapper, Maze2DFirstExitWrapper

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def visualize_policy(env, model, num_episodes=10, deterministic=False, render=True):
    """
    Visualize the policy in env
    """
    # Ensure testing on same device
    total_ep_returns = []
    total_ep_lengths = []

    for _ in trange(num_episodes):
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


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value

def build_mlp(
    in_size: int,
    hid_sizes: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    squeeze_output=False,
    flatten_input=False,
    use_spectral_norm: bool = False,
) -> nn.Module:
    """Constructs a Torch MLP.

    Args:
        in_size: size of individual input vectors; input to the MLP will be of
            shape (batch_size, in_size).
        hid_sizes: sizes of hidden layers. If this is an empty iterable, then we build
            a linear function approximator.
        out_size: required size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that MLP
            output is of size (B,) instead of (B,1).
        flatten_input: should input be flattened along axes 1, 2, 3, …? Useful
            if you want to, e.g., process small images inputs with an MLP.

    Returns:
        nn.Module: an MLP mapping from inputs of size (batch_size, in_size) to
            (batch_size, out_size), unless out_size=1 and squeeze_output=True,
            in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    from collections import OrderedDict
    layers = OrderedDict()

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    if flatten_input:
        layers[f"{prefix}flatten"] = nn.Flatten()

    # Hidden layers
    prev_size = in_size
    for i, size in enumerate(hid_sizes):
        if use_spectral_norm:
            layers[f"{prefix}dense{i}_SN"] = spectral_norm(nn.Linear(prev_size, size))
        else:
            layers[f"{prefix}dense{i}"] = nn.Linear(prev_size, size)
            
        prev_size = size
        if activation:
            layers[f"{prefix}act{i}"] = activation()

    # Final layer
    layers[f"{prefix}dense_final"] = nn.Linear(prev_size, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(layers)

    return model

class SpectralNormRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        use_spectral_norm=True,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs = {
            "hid_sizes": (32, 32),
        }
        full_build_mlp_kwargs.update(kwargs)
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size": combined_size,
                "out_size": 1,
                "squeeze_output": True,
                "use_spectral_norm": use_spectral_norm,
            },
        )

        self.mlp = build_mlp(**full_build_mlp_kwargs)

    def forward(self, state, action, next_state, done):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat)
        assert outputs.shape == state.shape[:1]

        return outputs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, choices=["bc", "airl", "gail"])
    p.add_argument("--train", action="store_true")
    p.add_argument("--render","-r", action="store_true")
    # p.add_argument("--num_steps", "-n ", type=int, default=10)
    p.add_argument("--fix_horizon", action="store_true")
    p.add_argument("--timelimit", action="store_true")
    p.add_argument("--noisy", action="store_true")
    args = p.parse_args()
    
    cwd = pathlib.Path(__file__).resolve().parent
    
    sb3_trajs = dict(
        point = "sb3_generated_trajectories/maze2d_traj.pkl",
        point_timelimit = "sb3_generated_trajectories/maze2d_timelimit_traj.pkl",
        point_noisy = "sb3_generated_trajectories/maze2d_noisy_traj.pkl",
        point_noisy_timelimit = "sb3_generated_trajectories/maze2d_noisy_timelimit_traj.pkl",
    )
    if args.timelimit:
        if args.noisy:
            fname = sb3_trajs["point_noisy_timelimit"]
            save_name = "noisy_timelimit"
        else:
            fname = sb3_trajs["point_timelimit"]
            save_name = "timelimit"
    else:
        if args.noisy:
            fname = sb3_trajs["point_noisy"]
            save_name = "noisy"
        else:
            fname = sb3_trajs["point"]
            save_name = "none"
    
    policy_fname = f"./{args.algo}_policy_" + save_name 
    expert_traj = cwd / "generated_dataset" / fname
    
    ic(args)
    print("Loading expert trajectory from {}".format(expert_traj))
    
    
    ISO_TIMESTAMP = "%Y%m%d_%H%M_%S"
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    if args.train:
        # Load pickled test demonstrations.
        with open(expert_traj, "rb") as f:
            # This is a list of `imitation.data.types.Trajectory`, where
            # every instance contains observations and actions for a single expert
            # demonstration.
            trajectories = pickle.load(f)

        # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
        # This is a more general dataclass containing unordered
        # (observation, actions, next_observation) transitions.
        transitions = rollout.flatten_trajectories(trajectories)
       
        data_dir = "./data"
        data_dir_path = pathlib.Path(data_dir).resolve()
        data_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"All Tensorboards and logging are being written inside {data_dir_path}/.")
        
        hp = dict(
            gen_lr = 5e-4,
            disc_lr = 7e-4,
            n_envs = 8,
            demo_batch_size = 256,
            n_disc_updates_per_round = 1,
        )
        hp["gen_algo_batch_size"] = int(1024 * hp["n_envs"])
        
        wrappers = [Maze2DFixedStartWrapper,] 
        if not args.timelimit:
            wrappers.append(Maze2DFirstExitWrapper)
        venv = util.make_vec_env("maze2d-umaze-v1", n_envs=hp["n_envs"], post_wrappers=wrappers)
        ic(wrappers)
        
        
        if args.algo == "bc":
            # Train BC on expert data.
            bc_logger = logger.configure(data_dir_path / "BC/", ["stdout", "csv", "tensorboard"])
            bc_trainer = bc.BC(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                demonstrations=transitions,
                custom_logger=bc_logger,
            )
            bc_trainer.train(n_epochs=500)
            bc_trainer.save_policy(policy_fname)
            print(f"BC policy saved to {policy_fname}")
        else:
            # *Reward Net
            SNRewardNet = SpectralNormRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                use_state=True,
                use_action=True,
                use_next_state=False,
                use_done=False,  # "done" flag be included as an input to the MLP?
                use_spectral_norm=True,
                hid_sizes=(64, 64),  # default: (32, 32)
            )
            
            if args.algo == "gail":
                # Train GAIL on expert data.
                # GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
                # iterates over dictionaries containing observations, actions, and next_observations.
                gail_logger = logger.configure(data_dir_path / f"GAIL/{timestamp}", ["stdout", "csv", "tensorboard"])
                gen_algo = sb3.PPO(
                        "MlpPolicy",
                        venv,
                        verbose=1,
                        n_steps=1024,
                        learning_rate=hp["gen_lr"],
                        batch_size=hp["gen_algo_batch_size"],  # size `n_steps * n_envs 
                )
                
                gail_trainer = gail.GAIL(
                    venv=venv,
                    demonstrations=transitions,
                    demo_batch_size=hp["demo_batch_size"],
                    gen_algo=gen_algo,
                    custom_logger=gail_logger,
                    disc_opt_cls = th.optim.Adam,
                    disc_opt_kwargs = {"lr": hp["disc_lr"]},
                    allow_variable_horizon=not args.fix_horizon,
                    n_disc_updates_per_round=hp["n_disc_updates_per_round"],
                    # * this part is important
                    normalize_obs=False,
                    normalize_reward= False,
                    # reward_net=gailRewardNet
                    model_save_dir = data_dir_path / f"GAIL/{timestamp}" / policy_fname
                )
                gail_trainer.train(total_timesteps=int(6e6))
                gen_algo.save(data_dir_path / f"GAIL/{timestamp}/policy")
                gen_algo.save(policy_fname)
                
                print(f"GAIL policy saved to {policy_fname}")
            
            elif args.algo == "airl":
                # Train AIRL on expert data.
                airl_logger = logger.configure(data_dir_path / f"AIRL/{timestamp}", ["stdout", "csv", "tensorboard"])
                gen_algo = sb3.PPO(
                        "MlpPolicy",
                        venv,
                        verbose=1,
                        n_steps=1024,
                        learning_rate=hp["gen_lr"],
                        batch_size=hp["gen_algo_batch_size"],  # size `n_steps * n_envs 
                )
                
                airl_trainer = airl.AIRL(
                    venv=venv,
                    demonstrations=transitions,
                    demo_batch_size=hp["demo_batch_size"],
                    gen_algo=gen_algo,
                    custom_logger=airl_logger,
                    allow_variable_horizon=not args.fix_horizon,
                    disc_opt_cls = th.optim.Adam,
                    disc_opt_kwargs = {"lr": hp["disc_lr"]},
                    n_disc_updates_per_round=hp["n_disc_updates_per_round"],
                    normalize_obs=False,
                    normalize_reward= False,
                    model_save_dir = data_dir_path / f"AIRL/{timestamp}" / policy_fname
                    
                )
                airl_trainer.train(total_timesteps=int(7e6))
                gen_algo.save(data_dir_path / f"AIRL/{timestamp}/policy")
                gen_algo.save(policy_fname)
                print(f"AIRL policy saved to {policy_fname}")
        
            with open(os.path.join(data_dir_path / f"{args.algo.upper()}/{timestamp}", "hyperparams.yaml"), "w") as f:
                yaml.dump(hp, f)
     
    if args.train:
        policy = gen_algo if args.algo in ["gail, aril"] else bc_trainer.policy
    else:
        # load trained policy
        policy = sb3.PPO.load(policy_fname) #if args.algo in ["gail, aril"] else th.load(f"{policy_fname}.pth", map_location=th.device('cpu'))
    
    test_wrappers = [Maze2DFixedStartWrapper,] 
    if not args.timelimit:
        test_wrappers.append(Maze2DFirstExitWrapper)
    test_env = util.make_vec_env("maze2d-umaze-v1", n_envs=1, post_wrappers=test_wrappers)
    
        
    visualize_policy(test_env, policy, num_episodes=50, deterministic=False, render=args.render)
    
    # 50 episode            mean reward +- std
    #                       time limit              First exit
    # Expert(waypoint)
    # BC                    138.88 +/- 0.0          123.64
    # GAIL                  166.16 +/- 0.0          0.0     (close) see render
    # AIRL                  121.76 +/- 0.0          1.00 +/- 5.2
    # BCQ
    # CQL
    
    
    # regenerate dataset with rnd noise
    #                       time limit              First exit
    # Expert(waypoint)
    # BC                                               -
    # GAIL                  112.42 +/- 0.0           0.0
    # AIRL                  108.94 +/- 0.0           0.0
    # BCQ
    # CQL

    


    

    
