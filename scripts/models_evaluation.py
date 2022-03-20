from stable_baselines3 import PPO, DQN
import gym
import highway_env
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import time
from configs import get_config
from highway_env.road.my_image_renderer import MyImageRenderer
import matplotlib.pyplot as plt
import cv2 as cv
from datetime import datetime
import os


if __name__ == "__main__":
    #model = PPO.load("../models/PPO_06-03-2022")
    #model = PPO.load(r"D:\projects\RL\highway-env\models\PPO_07-03-2022.zip")
    model = PPO.load(r"D:\projects\RL\highway-env\models\PPO_16-03-2022\no_7.zip")
    model = DQN.load(r"D:\projects\RL\highway-env\models\DQN_17-03-2022\DQN_2000000_steps.zip")
    Save = False
    datetimestr = datetime.now().strftime('%d-%m-%Y')
    image_folder = r"D:\projects\RL\highway-env\results\\"+datetimestr+r"\\"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    Show = True
    #env = gym.make("highway-fast-v0", config=config)
    #env = highway_env.envs.HighwayEnvFast(config)
    env = highway_env.envs.HighwayEnvFast(get_config(is_test=False))
    env.seed(1234)
    env.reset()
    myImageRenderer = MyImageRenderer(env)
    n_collisions = 0
    n_off_road = 0
    n_success = 0
    n_scenes = 1000
    mean_speed = 0
    cnt = 0
    cnt_mean_speed_close_to_target_5_pct = 0
    cnt_mean_speed_close_to_target_10_pct = 0
    speeds = []
    i = 0
    for _ in range(n_scenes):
        start_reset = time.perf_counter()
        obs = env.reset()
        end_reset = time.perf_counter()
        #print('Time to reset: ',end_reset - start_reset)
        done = False
        step_times = []
        target_speeds = [v.target_speed for v in env.road.vehicles]
        #print(env.road.vehicles[2].target_speed,env.road.vehicles[2].speed)
        speeds = [v.speed for v in env.road.vehicles]
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            ego_speed = env.controlled_vehicles[0].speed
            #print('action = {}, speed = {}, lane = {}'.format(action, ego_speed,env.vehicle.target_lane_index[2]))
            #print(env.road.vehicles[2].speed)
            mean_speed = float(cnt)/(cnt+1)*mean_speed + 1.0/(cnt + 1)*ego_speed
            cnt += 1
            start_step = time.perf_counter()
            obs, reward, done, info = env.step(action)
            end_step = time.perf_counter()
            #print('step time = ', end_step - start_step)
            # new_speeds = [v.speed for v in env.road.vehicles]
            # for i, (s_o, s_n) in enumerate(zip(new_speeds, speeds)):
            #     if s_n - s_o < -(1.0 / 3) * 3.5:
            #         print('v {} decelarted from {} to {}'.format(i, s_o, s_n))
            # speeds = new_speeds
            step_times.append(end_step - start_step)
            if env.vehicle.crashed:
                n_collisions += 1
            if env.steps >= env.config["duration"] and not env.vehicle.crashed:
                n_success += 1
            if env.config["offroad_terminal"] and not env.vehicle.on_road:
                n_off_road += 1
            # start = time.perf_counter()
            image = myImageRenderer.my_render()
            # plt.pause(0.0001)
            # end = time.perf_counter()
            # print('rendering: ',end-start)
            if Show:
                # plt.imshow(image)
                # plt.pause(0.001)
                # time.sleep(0.01)
                env.render()
                time.sleep(0.01)
            if Save:
                imname = str(i)
                while len(imname) < 4:
                    imname = '0' + imname
                image = np.dstack(((image * 150) % 255, (image * 333) % 255, (image * 98) % 255)).astype(np.uint8)
                #image = cv.resize(image, (160, 480), interpolation=cv.INTER_NEAREST)
                # cv.imwrite(r"D:\projects\RL\Avatar\log\images\result\\" + imname + ".png", image)
                cv.imwrite(image_folder + imname + ".png", image)
                i += 1

            # if done:
            #     myImageRenderer.reset_pos()
            #     print("done")
            #for i,v in enumerate(env.road.vehicles):
            #    if i > 0 and v.speed > target_speeds[i]:
            #        print('increased from {} to {} at {}'.format(target_speeds[i],v.speed,i))




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