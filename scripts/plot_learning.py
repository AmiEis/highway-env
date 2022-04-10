from stable_baselines3 import PPO, DQN
import gym
import highway_env
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt
from configs import get_config
from os import listdir
from os.path import join, exists
from os import makedirs

env = highway_env.envs.HighwayEnvFast(get_config(is_test=False))
action_type = 'DiscreteAction'
env.config["action"]["type"] = action_type
n_scenes = 200
success_rate = []
km_per_collision = []
mean_speeds = []
ego_speed_at_5pct = []
ego_speed_at_10pct = []
emergency_breaks_pct = []

if __name__ == "__main__":
    base_dir = r"D:\projects\RL\highway-env\models\PPO_07-04-2022"
    datestr = base_dir.split('\\')[-1].split('_')[-1]
    image_folder = r"D:\projects\RL\highway-env\results\graphs\\" + datestr + r"\\"
    if not exists(image_folder):
        makedirs(image_folder)
    model_files = listdir(base_dir)
    model_files.sort(key=lambda fn: int(fn.split('.')[0].split('_')[1]))

    for i,fn in enumerate(model_files):
        model = PPO.load(join(base_dir, fn))
        env.seed(1234)
        env.reset()
        n_collisions = 0
        n_off_road = 0
        n_success = 0
        mean_speed = 0
        cnt = 0
        cnt_mean_speed_close_to_target_5_pct = 0
        cnt_mean_speed_close_to_target_10_pct = 0
        speeds = []
        lane_change = False
        lane_changes_n = 0
        rear_breaking_n = 0
        distance_passed = 0
        for _ in range(n_scenes):
            obs = env.reset()
            done = False
            pos_start = env.vehicle.position[0]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                ego_speed = env.controlled_vehicles[0].speed
                mean_speed = float(cnt) / (cnt + 1) * mean_speed + 1.0 / (cnt + 1) * ego_speed
                cnt += 1
                ego_vehicle_lane = env.vehicle.lane_index
                obs, reward, done, info = env.step(action)
                lane_change = ego_vehicle_lane != env.vehicle.lane_index
                if lane_change:
                    has_rear, rear_break = env.calc_rear_break(is_test=True)
                    if has_rear:
                        lane_changes_n += 1
                        if rear_break <= -5:
                            rear_breaking_n += 1
                if env.vehicle.crashed:
                    n_collisions += 1
                if env.steps >= env.config["duration"] and not env.vehicle.crashed:
                    n_success += 1
                if env.config["offroad_terminal"] and not env.vehicle.on_road:
                    n_off_road += 1
                if done:
                    pos_end = env.vehicle.position[0]
                    distance_passed += pos_end - pos_start
            #if np.abs(mean_speed - 30)/30 < 0.05:
            #    cnt_mean_speed_close_to_target_5_pct += 1
            #if np.abs(mean_speed - 30) / 30 < 0.1:
            #    cnt_mean_speed_close_to_target_10_pct += 1
        cnt = 0
        success_rate.append(float(n_success)/n_scenes)
        km_per_collision.append((distance_passed/1000)/max(float(n_collisions),1e-16))
        mean_speeds.append(mean_speed)
        #ego_speed_at_5pct.append(float(cnt_mean_speed_close_to_target_5_pct)/n_scenes)
        #ego_speed_at_10pct.append(float(cnt_mean_speed_close_to_target_10_pct)/n_scenes)
        emergency_breaks_pct.append(float(rear_breaking_n)/max(float(lane_changes_n),1e-16))
    ran = range(1,len(listdir(base_dir))+1)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(ran, success_rate)
    axs[0].set_title('Success rate %')
    axs[0].set(xlabel='M learning steps', ylabel='Success rate')
    axs[1].plot(ran, km_per_collision)
    axs[1].set_title('KM per Collision')
    axs[1].set(xlabel='M learning steps', ylabel='KM per Collision')
    #axs[2].plot(ran, emergency_breaks_pct)
    #axs[2].set_title('Emergency Breaks % at Lane Change')
    #axs[2].set(xlabel='M learning steps', ylabel='Emergency Breaks')
    fig.tight_layout()
    plt.savefig(join(image_folder,'fig1.png'))
    fig, axs = plt.subplots(2,1)
    axs[0].plot(ran, emergency_breaks_pct)
    axs[0].set_title('Emergency Breaks % at Lane Change')
    axs[0].set(xlabel='M learning steps', ylabel='Success rate')
    axs[1].plot(ran, mean_speeds)
    axs[1].set_title('Mean Speed (target speed = 30)')
    axs[1].set(xlabel='M learning steps', ylabel='mean speed')
    #axs[0].plot(ran, ego_speed_at_5pct)
    #axs[0].set_title('Rate at 5% of target speed')
    #axs[0].set(xlabel='M learning steps', ylabel='at 5% of target speed')
    #axs[1].plot(ran, ego_speed_at_10pct)
    #axs[1].set_title('Rate at 10% of target speed')
    #axs[1].set(xlabel='M learning steps', ylabel='at 10% of target speed')
    fig.tight_layout()
    plt.savefig(join(image_folder,'fig2.png'))
    lines = []
    lines.append(','.join(map(str, success_rate)))
    lines.append(','.join(map(str, km_per_collision)))
    lines.append(','.join(map(str, emergency_breaks_pct)))
    lines.append(','.join(map(str, mean_speeds)))
    lines = map(lambda l: l + '\n',lines)
    with open(join(image_folder,'results.txt'),'w') as file:
        file.writelines(lines)
