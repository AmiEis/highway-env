import gym
import highway_env.envs.highway_env_scene
from highway_env.envs.highway_env_scene import HighwayEnvFast
import sys
from configs import get_config
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from highway_env.road.my_image_renderer import MyImageRenderer





if __name__ == "__main__":
    #env = gym.make("highway-v0")
    #config = {"action":{"type":"DiscreteAction"}}
    #config = {"action": {"type": "ContinuousAction"}}
    env = HighwayEnvFast(get_config(is_test=False))
    env.config["action"] = {"type":"DiscreteAction", "actions_per_axis":3}
    env.config["vehicles_density"] = 1
    myImageRenderer = MyImageRenderer(env)
    obs = env.reset()
    #env.render()
    for i in range(10000):
        '''if i % 10 == 0:
            action = 1
        elif i % 20 == 0:
            action = 2
        else:
            action = 1'''

        '''if i % 5 == 0:
            action = (0,0.1)
        elif i % 5 == 1:
            action = (0,-0.1)
        elif i % 5 == 2:
            action = (0,-0.1)
        elif i % 5 == 3:
            action = (0,0.1)
        else:
            action = (0,0)'''
        action = 4
        # if i % 100 == 0:
        #    action = np.random.randint(9)


         # if i % 100 == 0:
         #    s = np.random.randint(2)
         #    action = 0 if s == 0 else 2
        obs, rew, done, info = env.step(action)
        env.render()
        myImageRenderer.my_render()
        plt.pause(0.0001)
        '''speeds = [v.speed for v in env.road.vehicles]
        target_speeds = [v.target_speed for v in env.road.vehicles]
        positions = [v.position[0] for v in env.road.vehicles]
        print(speeds)
        print(target_speeds)
        print(positions)'''
        if done:
            env.reset()
            print('done')


