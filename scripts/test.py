import gym
import highway_env.envs.highway_env_scene
from highway_env.envs.highway_env_scene import HighwayEnvFast
import sys
from configs import get_config

if __name__ == "__main__":
    #env = gym.make("highway-v0")
    #config = {"action":{"type":"DiscreteAction"}}
    #config = {"action": {"type": "ContinuousAction"}}
    env = HighwayEnvFast(get_config())
    env.config["lanes_count"] = 3
    obs = env.reset()
    for i in range(1000):
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
        action = 1
        obs, rew, done, info = env.step(action)
        env.render()
        print(env)
        if done: env.reset()


