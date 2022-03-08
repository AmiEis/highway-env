from stable_baselines3 import PPO
import gym
import highway_env
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import time
from configs import get_config


if __name__ == "__main__":
    #model = PPO.load("../models/PPO_06-03-2022")
    model = PPO.load(r"D:\projects\RL\highway-env\models\PPO_07-03-2022.zip")

    #env = gym.make("highway-fast-v0", config=config)
    #env = highway_env.envs.HighwayEnvFast(config)
    env = highway_env.envs.HighwayEnvFast(get_config(is_test=True))
    env.seed(1234)
    env.reset()
    n_collisions = 0
    n_off_road = 0
    n_success = 0
    n_scenes = 10
    mean_speed = 0
    cnt = 0
    cnt_mean_speed_close_to_target_5_pct = 0
    cnt_mean_speed_close_to_target_10_pct = 0
    speeds = []
    for _ in range(n_scenes):
        start_reset = time.perf_counter()
        obs = env.reset()
        end_reset = time.perf_counter()
        #print('Time to reset: ',end_reset - start_reset)
        done = False
        step_times = []
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            ego_speed = env.controlled_vehicles[0].speed
            mean_speed = float(cnt)/(cnt+1)*mean_speed + 1.0/(cnt + 1)*ego_speed
            cnt += 1
            start_step = time.perf_counter()
            obs, reward, done, info = env.step(action)
            end_step = time.perf_counter()
            #print('step time = ', end_step - start_step)
            step_times.append(end_step - start_step)
            if env.vehicle.crashed:
                n_collisions += 1
            if env.steps >= env.config["duration"] and not env.vehicle.crashed:
                n_success += 1
            if env.config["offroad_terminal"] and not env.vehicle.on_road:
                n_off_road += 1
            env.render()
            for i,v in enumerate(env.road.vehicles):
                if len(speeds)>0:
                    if i > 0 and v.speed > speeds[i]:
                        print('increased from {} to {} at {}'.format(speeds[i],v.speed,i))
            speeds = []
            for v in env.road.vehicles:
                speeds.append(v.speed)



        #print("episode avg speed = ",mean_speed)
        #print('Mean step time for episode: ',np.mean(step_times))
        #print('Total step time for episode: ',np.sum(step_times))
        if np.abs(mean_speed - 30)/30 < 0.05:
            cnt_mean_speed_close_to_target_5_pct += 1
        if np.abs(mean_speed - 30) / 30 < 0.1:
            cnt_mean_speed_close_to_target_10_pct += 1
        cnt = 0
    print("Sucess rate = ",float(n_success)/n_scenes)
    print("Num collisions: {} out of {} scenes".format(n_collisions,n_scenes))
    print("Rate Ego speed at 5% of target speed: ",float(cnt_mean_speed_close_to_target_5_pct)/n_scenes)
    print("Rate Ego speed at 10% of target speed: ",float(cnt_mean_speed_close_to_target_10_pct)/n_scenes)