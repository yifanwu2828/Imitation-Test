#!/bin/bash


source venv/bin/activate

pip install --upgrade pip wheel setuptools

pip install numpy

pip install icecream

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -U 'mujoco-py<2.2,>=2.1'

pip install git+https://github.com/DLR-RM/stable-baselines3

pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

pip install dm-control==0.0.403778684

