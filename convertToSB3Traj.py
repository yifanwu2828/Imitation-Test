import argparse
import pathlib
from typing import Optional
import pickle
from click import parser

import numpy as np
import h5py

from icecream import ic

from imitation.data.types import Trajectory



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--env", type=str, required=True, choices=["ant", "maze2d"])
    parser.add_argument("--timelimit", action="store_true")
    parser.add_argument("--noisy", action="store_true")
    args = parser.parse_args()
    
    key = f"{args.env}_noisy" if args.noisy else f"{args.env}"
    key = f"{key}_timelimit" if args.timelimit else key 
    
    pwd = pathlib.Path(__file__).parent.resolve()
    
    file_map = dict(
        maze2d = pwd /'generated_dataset/maze2d-umaze-v1.hdf5',
        maze2d_timelimit = pwd /'generated_dataset/maze2d-umaze-v1_timelimit.hdf5',
        maze2d_noisy = pwd /"generated_dataset/maze2d-umaze-v1-noisy.hdf5",
        maze2d_noisy_timelimit = pwd / "generated_dataset/maze2d-umaze-v1_timelimit-noisy.hdf5"
    )
    
    dataset_file = h5py.File(file_map[key], 'r')
    
    data = dict(
        act = np.asarray(dataset_file['actions']),
        obs = np.asarray(dataset_file['observations']),
        terminal = np.asarray(dataset_file['terminals']),
        # timeout = np.asarray(f['timeouts']),
    )
    
    obs_dim = data['obs'].shape[1]
    act_dim = data['act'].shape[1]
    
    
    trajectories = [] 

    if args.env == "maze2d":
        # all dones include s_T and s_{T+1}
        # [299,300,600,601,901,902,1202,1203]
        dones = [i for i in range(data['obs'].shape[0]) if data['terminal'][i]]
        assert len(dones) / 2 == 100, "number of traj is not 100"
        ic(dones)
        # take odd idx which is the index of s_{T+1} of each trajectory
        # [300,601,902,1203]
        dones = [d for idx, d in enumerate(dones) if idx % 2 != 0] 
        assert len(dones) == 100
        ic(dones) 
        i = 0
        for num, j in enumerate(dones):
            
            # i should be the index of start of each trajectory 
            assert data["obs"][i, 0] == 3 and data["obs"][i,1] == 1, f"start position is not (3,1): {data['obs'][i, :2]}"
            # j should be the index of s_{T+1} of each trajectory
            
            traj = Trajectory(
                obs=data["obs"][i:j+1, :],
                acts=data["act"][i:j, :],
                terminal=True,
                infos=None
            )
            trajectories.append(traj)
            # update the index of start of next trajectory
            # eg [299,300,   600,601,] -> [301, 602,]
            i = j + 1
   
            
    
    path = f"generated_dataset/sb3_generated_trajectories/{key}_traj.pkl"
    p = pathlib.Path(path)
    p.parent.mkdir(exist_ok=True, parents=True)
    
    # tmp_path = f"{path}.tmp"
    # with open(tmp_path, "wb") as f:
    with open(path, "wb") as f:
        pickle.dump(trajectories, f)
    # Ensure atomic write
    # os.replace(tmp_path, f"{path}.pkl")
    print(f"Dumped demonstrations to {path}.")