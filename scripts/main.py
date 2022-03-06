import gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
from stable_baselines3.common.vec_env import VecFrameStack
from datetime import datetime
import os
from highway_env.envs.highway_env_scene import HighwayEnvFast

n_envs = 16
n_stack = 4
tensorboard_log = "highway_ppo/"
Alg = "PPO"

if __name__ == "__main__":
    config = {"config":{
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    }}
    env = make_vec_env(HighwayEnvFast,n_envs=16,env_kwargs=config)
    model = PPO('CnnPolicy',
                env,
                verbose=0,
                n_steps=512,
                target_kl=0.015,
                gamma=0.99,
                gae_lambda=0.9,
                ent_coef=0.015,
                tensorboard_log=tensorboard_log,
                device='cuda:0')

    datetimestr = datetime.now().strftime('%d-%m-%Y')
    #checkpoint_dir = Alg + "_" + datetimestr
    #base_dir = os.path.join(r"D:\projects\RL\highway-env\models", checkpoint_dir)
    models_dir = r"D:\projects\RL\highway-env\models"
    model.learn(2_000_000)
    model.save(os.path.join(models_dir,Alg+"_"+datetimestr))
    #if not os.path.exists(base_dir):
    #    os.makedirs(base_dir)
    '''for i in range(8):
        model.learn(1_000_000, reset_num_timesteps=True if i == 0 else False)
        model.save(os.path.join(base_dir, "no_" + str(i)))
        '''
