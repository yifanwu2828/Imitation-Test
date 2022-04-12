import argparse
import pathlib

from d3rlpy.dataset import MDPDataset
import h5py
import numpy as np
from sklearn import datasets


if __name__  == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timelimit", action="store_true")
    p.add_argument("--noisy", action="store_true")
    args = p.parse_args()
    
    cwd = pathlib.Path(__file__).resolve().parent
    custom_dataset = dict(
        point = "maze2d-umaze-v1",
        point_timelimit = "maze2d-umaze-v1_timelimit",
        point_noisy = "maze2d-umaze-v1-noisy",
        point_noisy_timelimit = "maze2d-umaze-v1_timelimit-noisy",
    )
    if args.timelimit:
        if args.noisy:
            fname = custom_dataset["point_noisy_timelimit"]
        else:
            fname = custom_dataset["point_timelimit"]
    else:
        if args.noisy:
            fname = custom_dataset["point_noisy"]
        else:
            fname = custom_dataset["point"]
            
    data_dir = cwd / "generated_dataset"
    print(f"Loading {data_dir / fname}.hdf5")
    f = h5py.File(f"{data_dir / fname}.hdf5", 'r')

    observations = np.asarray(f['observations']) 
    actions = np.asarray(f['actions'])
    rewards = np.asarray(f['rewards'])
    terminals = np.asarray(f['terminals'])
    
    
    custom_dataset = MDPDataset(observations, actions, rewards, terminals)
    
    # save as HDF5
    d3rl_MDP_dir = cwd / "generated_dataset" / "d3rl_MDPDataset"
    save_path = d3rl_MDP_dir / f"d3rl_{fname}.h5"
    custom_dataset.dump( save_path)
    print(f"Saving to {save_path}")
    
    # load from HDF5
    new_dataset = MDPDataset.load(save_path)
    print("Load check succeed!")