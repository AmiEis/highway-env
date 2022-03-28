from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from datetime import datetime
import os
from highway_env.envs.highway_env_scene import HighwayEnvFast
from configs import get_config
from sys import exit
from stable_baselines3.common.callbacks import CheckpointCallback

n_envs = 16
alg = 'DQN'
tensorboard_log = "highway_{}/".format(alg.lower())
save_freq = 1_000_000

if __name__ == "__main__":
    config = {"config": get_config()}
    if alg == 'PPO':
        save_freq = max(save_freq // n_envs, 1)
        env = make_vec_env(HighwayEnvFast, n_envs=16, env_kwargs=config)
        model = PPO('CnnPolicy',
                    env,
                    verbose=0,
                    n_steps=512,
                    target_kl=0.015,
                    gamma=0.95,
                    gae_lambda=0.9,
                    ent_coef=0.015,
                    tensorboard_log=tensorboard_log,
                    device='cuda:0')
    elif alg == 'DQN':

        model = DQN('CnnPolicy',
                    DummyVecEnv([lambda: HighwayEnvFast(config=get_config())]),
                    learning_rate=1e-4,
                    buffer_size=300_000,
                    learning_starts=1000,
                    batch_size=32,
                    gamma=0.95,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    exploration_fraction=0.4,
                    verbose=0,
                    tensorboard_log=tensorboard_log)
    else:
        exit("Alg \'{}\' not supported!".format(alg))


    datetimestr = datetime.now().strftime('%d-%m-%Y')
    #model = PPO.load(r"D:\projects\RL\highway-env\models\PPO_16-03-2022\no_0.zip",env)
    #checkpoint_dir = Alg + "_" + datetimestr
    #base_dir = os.path.join(r"D:\projects\RL\highway-env\models", checkpoint_dir)
    models_dir = os.path.join("D:\projects\RL\highway-env\models",alg+"_"+datetimestr)
    if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    # Save a checkpoint every save_freq steps
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=models_dir,
                                             name_prefix=alg)
    model.learn(8_000_000,callback=checkpoint_callback)
    # for i in range(1,8):
    #     model.learn(1_000_000,reset_num_timesteps=True if i==0 else False)
    #     model.save(os.path.join(models_dir, "no_" + str(i)))
    #model.learn(2_000_000)
    #model.save(os.path.join(models_dir,Alg+"_"+datetimestr))

