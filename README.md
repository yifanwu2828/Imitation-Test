# Imitation-Test

## Install
  - run requirment.sh
  - All scripts are placed at the root level.

## Training Scriot
  - train_irl_expert.py (SB3)
  - train_offline.py (d3rlpy)

## Model
  - sb3 model are save with `.zip` extension
  - d3rl model are saved with `.pt` extension

## Utility
  - `convertToSB3Traj.py` work with sb3 imitation
  - `convertMDPDataset.py` work with d3rlpy.
  - `wrappers.py to fix` the start position and goal position
  - `dict_obs_wapper.py` only convert the observation to dict but not the observation space
  - `goalEnvWrapper.py` TBD. Not working correctly
  - `visualize_traj.ipynb` to view expert demostration.

###
Maze 2D can be trained in RL with SAC. If fixed the start position, it won't work.
