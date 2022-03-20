import gym
import highway_env.envs.highway_env_scene
from highway_env.envs.highway_env_scene import HighwayEnvFast
import sys
from configs import get_config
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from highway_env.road.my_image_renderer import MyImageRenderer
import random
import time
from highway_env.envs.lane_keeping_env import LaneKeepingEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.envs.narrow_pass import NarrowPass

if __name__ == "__main__":
    #env = gym.make("highway-v0")
    #config = {"action":{"type":"DiscreteAction"}}
    #config = {"action": {"type": "ContinuousAction"}}
    env = HighwayEnvFast(get_config(is_test=False))
    #env.config["action"] = {"type":"DiscreteAction", "actions_per_axis":3}
    env.config["vehicles_density"] = 1
    env.config["screen_width"] = 256
    env.config["screen_height"] = 64
    env.config["scaling"] = 2.5
    env = NarrowPass({
        "observation": {
            "type": "GrayscaleObservation",
            #"type": "MyImageObservation",
            "observation_shape": (256, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 2.5,
        },
    })
    obs = env.reset()
    #myImageRenderer = MyImageRenderer(env)
    env.render()
    for i in range(10000):
        action = 1#[0,0]#random.choice([1,3,4])
        # if i % 10 == 0: action = random.choice([0,2])
        obs, rew, done, info = env.step(action)
        env.render()
        #myImageRenderer.my_render()
        time.sleep(0.2)
        plt.pause(0.0001)
        '''target_speeds = [v.target_speed for v in env.road.vehicles]
        positions = [v.position[0] for v in env.road.vehicles]
        print(speeds)
        print(target_speeds)
        print(positions)'''
        if done:
            env.reset()
            # myImageRenderer.reset_pos()
            print('done')


